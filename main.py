# Downloaded on 2023-08-21 from https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/main.py

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Optional, cast, Dict, Any
from pathlib import Path

import src.mosaic_bert as mosaic_bert_module
import src.text_data as text_data_module
from composer import Trainer, algorithms
from composer.callbacks import HealthChecker, LRMonitor, MemoryMonitor, OptimizerMonitor, RuntimeEstimator, SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import ConstantWithWarmupScheduler, CosineAnnealingWithWarmupScheduler, LinearWithWarmupScheduler
from composer.utils import dist, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from huggingface_hub import login, logout, HfApi
from pandas import DataFrame
from dotenv import load_dotenv
from flash_attention_softmax_n.analysis import register_activation_hooks, compute_weight_statistics, save_results


def update_batch_size_info(cfg: DictConfig):
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} '
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            f'to be divisible by world size, {dist.get_world_size()}.')
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f'WARNING: device_train_microbatch_size > device_train_batch_size, '
                f'will be reduced from {device_microbatch_size} -> {device_train_batch_size}.'
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg.device_train_microbatch_size == 'auto':
            cfg.device_eval_batch_size = 1
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


def build_algorithm(name, kwargs):
    if name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    elif name == 'alibi':
        return algorithms.Alibi(**kwargs)
    elif name == 'fused_layernorm':
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == 'gated_linear_units':
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == 'low_precision_layernorm':
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')


def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1),
                            gpu_flops_available=kwargs.get('gpu_flops_available', None))
    elif name == 'runtime_estimator':
        return RuntimeEstimator()
    elif name == 'optimizer_monitor':
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get('log_optimizer_metrics', True),)
    elif name == 'health_checker':
        return HealthChecker(**kwargs)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')


def build_optimizer(cfg, model):
    if cfg.name == 'decoupled_adamw':
        return DecoupledAdamW(model.parameters(),
                              lr=cfg.lr,
                              betas=cfg.betas,
                              eps=cfg.eps,
                              weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')


def build_dataloader(cfg, tokenizer, device_batch_size):
    if cfg.name == 'text':
        return text_data_module.build_text_dataloader(cfg, tokenizer, device_batch_size)
    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')


def build_model(cfg: DictConfig):
    if cfg.name == 'mosaic_bert':
        return mosaic_bert_module.create_mosaic_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get('pretrained_checkpoint', None),
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None),
            gradient_checkpointing=cfg.get('gradient_checkpointing', None))
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def summarize_stats(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats_df = DataFrame.from_dict(stats, orient='index')
    stats_df = stats_df.reset_index(names='name')
    stats_df['name'] = stats_df['name'].str.replace(r'[0-9]+', '', regex=True)
    stats_summary_df = stats_df.groupby('name').agg(['mean', 'std'])
    stats_summary_df.columns = [' '.join(col).strip() for col in stats_summary_df.columns.values]
    return stats_summary_df.dropna().to_dict(orient='index')


def main(cfg: DictConfig,
         return_trainer: bool = False,
         do_train: bool = True) -> Optional[Trainer]:
    print('Training using config: ')
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Build Model
    print('Initializing model...')
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.4e}')

    activation_stats = register_activation_hooks(model)

    # Dataloaders
    print('Building train loader...')
    train_loader = build_dataloader(
        cfg.train_loader,
        model.tokenizer,
        cfg.global_train_batch_size // dist.get_world_size(),
    )
    print('Building eval loader...')
    global_eval_batch_size = cfg.get('global_eval_batch_size', cfg.global_train_batch_size)
    eval_loader = build_dataloader(
        cfg.eval_loader,
        model.tokenizer,
        global_eval_batch_size // dist.get_world_size(),
    )

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers', {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks', {}).items()
    ]

    # Algorithms
    """
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]
    """

    if cfg.get('run_name') is None:
        cfg.run_name = os.environ.get('COMPOSER_RUN_NAME', 'bert')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        # algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_subset_num_batches=cfg.get('train_subset_num_batches', -1),
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get('device', None),
        device_train_microbatch_size=cfg.get('device_train_microbatch_size', 'auto'),
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep', -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
        python_log_level=cfg.get('python_log_level', None),
    )

    print('Logging config...')
    log_config(cfg)

    if do_train:
        print('Starting training...')
        trainer.fit()

    weight_stats = compute_weight_statistics(model)

    activation_stats_summary = summarize_stats(activation_stats)
    weight_stats_summary = summarize_stats(weight_stats)

    n = int(cfg.model.model_config.softmax_n_param)
    model_name = f"mosaic-bert-softmax{n}"
    repo_id = f"{os.getenv('HUGGINGFACE_USER')}/{model_name}"
    results_dir = Path.cwd() / "results"

    save_results({
        'activations': activation_stats,
        'weights': weight_stats,
        'activations_summary': activation_stats_summary,
        'weights_summary': weight_stats_summary
    }, model_name)

    try:
        token = os.getenv("HUGGINGFACE_TOKEN")
        login(token=token)

        model.model.push_to_hub(repo_id=repo_id)

        api = HfApi()
        results_path = results_dir / f"{model_name}.json"
        api.upload_file(
            path_or_fileobj=results_path,
            path_in_repo="results.json",
            repo_id=repo_id,
        )
    except ValueError:
        model.model.save_pretrained(save_directory=results_dir)
    finally:
        logout()

    if return_trainer:
        return trainer


if __name__ == '__main__':
    load_dotenv()
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)
