#!/bin/bash
set -eo pipefail

# Make yourself a copy of C4
python src/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4
# Verify that the dataloader works
python src/text_data.py --local_path ./my-copy-c4

# Verify that pre-training runs correctly
composer main.py yamls/test/main.yaml train_loader.num_workers=4 eval_loader.num_workers=4
composer main.py yamls/test/main.yaml model.model_config.softmax_n_param=1 train_loader.num_workers=4 eval_loader.num_workers=4
