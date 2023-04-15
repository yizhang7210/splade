# SPLADE - README for MIDS W210 Capstone team

Nicholas Schantz, Saniya Lakka, Yi Zhang


## One Time AWS SageMaker setup
- Create a "Notebook Instance" in aws SageMaker (_not_ Studio or Studio Notebook)
- Set up the connected Git repository to be a [clone of naver's splade](https://github.com/yizhang7210/splade.git) repository
- Set up a lifecycle configuration for the notebook for "Start notebook" to be something like the following:
    ```sh
    #!/bin/bash

    set -e

    sudo -u ec2-user -i <<'EOF'
    unset SUDO_UID
    # Install a separate conda installation via Miniconda
    WORKING_DIR=/home/ec2-user/SageMaker/custom-miniconda
    mkdir -p "$WORKING_DIR"
    wget https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O "$WORKING_DIR/miniconda.sh"
    bash "$WORKING_DIR/miniconda.sh" -b -u -p "$WORKING_DIR/miniconda" 
    rm -rf "$WORKING_DIR/miniconda.sh"
    # Create a custom conda environment
    source "$WORKING_DIR/miniconda/bin/activate"
    #conda create --yes --name splade_env python=3.9
    #conda activate splade_env
    conda activate splade-ready
    pip install --quiet ipykernel
    echo "Adding the new env to list of conda env locations"
    cat << EOF >> /home/ec2-user/.condarc
    envs_dirs:
      - /home/ec2-user/SageMaker/custom-miniconda/miniconda/envs
      - /home/ec2-user/anaconda3/envs
    EOF
    ```
- Start the notebook and run the following:
    ```sh
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate pyotrch_p39
    conda env create -f conda_splade_env.yml
    cd /home/ec2-user/SageMaker/custom-miniconda/miniconda/envs
    conda create --prefix splade-ready --clone splade
    ```
    
## Ongoing Notebook setup
- Go to AWS SageMaker and start the notebook
- Click "Open JupyterLab" to get into the notebook's interface
- Run the following commands to set it up:
    ```
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate splade
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    git config --global user.name "Your Name"
    git config --global user.email "your email"
    ```
- To run the topic based filtering code, use the `main` git branch
- To run the tree base approaches, use the `binary-tree-approach` git branch


## Commands to run the experiments
We used the pre-trained `naver/splade-cocondenser-ensembledistil` model as the baseline without transfer learing.

- To use the existing pre-trained model to index the passages, do
```
SPLADE_CONFIG_NAME="config_default.yaml" python3 -m splade.index init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir=experiments/index-full-data/index
```

- To run a retrieval experiment based on the validation data, do
```
SPLADE_CONFIG_NAME="config_default.yaml" python3 -m splade.retrieve init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil config.pretrained_no_yamlconfig=true config.index_dir=experiments/index-full-data/index config.out_dir=experiments/index-full-data-topic-baseline/out
```

## Configurations

For the topic based approach, there are a few configuration toggles that can be changed in the code.
- in splade/splade/retrieve.py:
```
FILTER_BY_TOPIC = False  # Configures whether to use topic filters or baseline SPLADE

FILTER_TOPICS_BY_VALUE_THRESHOLD = False  # Whether to use value based topic threshold
PASSAGE_TOPIC_THRESHOLD = 0.1             # Threashold for passage topic scores - values below this will be set to 0
QUERY_TOPIC_THRESHOLD = 0.1               # Threashold for query topic scores - values below this will be set to 0

FILTER_TOPICS_BY_RANKING = False          # Whether to use rank based topic threshold
PASSAGE_TOPIC_RANK_THRESHOLD = 4          # For passage topic scores - only take the top this number of scores
QUERY_TOPIC_RANK_THRESHOLD = 2            # For query topic scores - only take the top this number of scores
```