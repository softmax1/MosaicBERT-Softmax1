# Downloaded on 2023-08-21 from https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/src/mosaic_bert.py

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Mosaic BERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

from typing import Optional

import src.bert_layers as bert_layers_module
import src.configuration_bert as configuration_bert_module
import transformers
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel


__all__ = ['create_mosaic_bert_mlm']


def create_mosaic_bert_mlm(pretrained_model_name: str = 'bert-base-uncased',
                           model_config: Optional[dict] = None,
                           tokenizer_name: Optional[str] = None,
                           gradient_checkpointing: Optional[bool] = False,
                           pretrained_checkpoint: Optional[str] = None):
    """Mosaic BERT masked language model based on |:hugging_face:| Transformers.

    For more information, see
    `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a Mosaic BERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided, the state dictionary
            stored at `pretrained_checkpoint` will be loaded into the model
            after initialization. Default: ``None``.

    .. code-block::

        {
        "_name_or_path": "bert-base-uncased",
        "alibi_starting_size": 512,
        "architectures": ["BertForMaskedLM"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": null,
        "gradient_checkpointing": false,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.16.0",
        "type_vocab_size": 2,
        "use_cache": true,
        "vocab_size": 30522
        }

    To create a Mosaic BERT model for Masked Language Model pretraining:

     .. testcode::

         from src.mosaic import create_mosaic_bert_mlm
         model = create_mosaic_bert_mlm()
    """
    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = 'bert-base-uncased'

    config = configuration_bert_module.BertConfig.from_pretrained(pretrained_model_name, **model_config)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = bert_layers_module.BertForMaskedLM.from_composer(pretrained_checkpoint=pretrained_checkpoint, config=config)
    else:
        model = bert_layers_module.BertForMaskedLM(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

    metrics = [
        LanguageCrossEntropy(ignore_index=-100),
        MaskedAccuracy(ignore_index=-100)
    ]

    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                use_logits=True,
                                metrics=metrics)

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model
