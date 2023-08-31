#!/bin/bash
set -eo pipefail

# We need to use the Git Large File System to save our model to Hugging Face hub
# These are AWS-specific instructions
sudo amazon-linux-extras install epel -y
sudo yum-config-manager --enable epel
sudo yum install git-lfs
git lfs install

# Clone the repo, and move into it
git clone https://github.com/softmax1/MosaicBERT-Softmax1.git
cd MosaicBERT-Softmax1/

# Activate the venv on AWS
source activate pytorch

# Install packages
pip install -r requirements.txt

# Make yourself a copy of C4. This step takes about 2 hours on a g5.2xlarge EC2 instance. The dataset is 125 GB in size.
python src/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train_medium val
# Verify that the dataloader works
python src/text_data.py --local_path ./my-copy-c4 --split val

# Verify that pre-training runs correctly
composer main.py yamls/test/main.yaml
composer main.py yamls/test/main.yaml model.model_config.softmax_n_param=1

# More testing
composer main.py yamls/main/mosaic-bert-uncased-tiny.yaml
composer main.py yamls/main/mosaic-bert-uncased-tiny.yaml model.model_config.softmax_n_param=1

# MLM pre-training
composer main.py yamls/main/mosaic-bert-uncased.yaml
composer main.py yamls/main/mosaic-bert-uncased.yaml model.model_config.softmax_n_param=1

# Wait for everything to finish, and deactivate the venv
wait
conda deactivate