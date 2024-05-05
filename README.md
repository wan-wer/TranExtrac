# Adaptive Sentence Selection for Extractive Summarization: Leveraging Dynamic Extraction Strategies

This repo contains the code, data and trained models.

## Quick Links

- [Overview](#overview)
- [How to Install](#how-to-install)
- [Description of Codes](#description-of-codes)
  - [Workspace](#workspace)
- [Data](#Data)
- [How to Run](#how-to-run)
  -  [Train](#train)
  -  [Evaluate](#evaluate)

## Overview

We propose a novel framework for extractive summarization that enables the model to automatically control the number of extracted sentences. 


## How to Install

- python 3.8

- pytorch 2.0.0

- transformers 4.31.0

- Further steps

    - ```
        pip install -U git+https://github.com/pltrdy/pyrouge
        git clone https://github.com/pltrdy/files2rouge.git     
        cd files2rouge
        python setup_rouge.py
        python setup.py install
        ```

## Description of Codes
- `/path/evaluate.py` -> evaluating procedure
- `/path/configuration_TranExtrac.py` -> model configuration
- `/path/train_TranExtrac.py` -> training procedure
- `/path/modeling_TranExtrac.py` -> model


### Workspace
Following directories should be created for our experiments.
- `./wandb` -> storing wandb information

## Data

Training Data could be downloaded through this link: [TranExtrac Google Drive](https://drive.google.com/drive/folders/1xx-9-EinBhhOdBn1TyCaRuORzDRCIqoX?usp=sharing).


## How to Run

We use an A100 to train our model.

### Train
```console
python /path/train_TranExtrac.py --train_data_path /path/train_data_path --val_data_path /path/val_data_path --pretrained_model_path /path/pretrained_bert_model  --model_save_path /path/model_save_path --num_train_epochs 15 --batch_size 48 --warmup_steps 8000 --eval_steps 400 --learning_rate 2e-5
```
The checkpoints and log will be saved in `/path/model_save_path`. We use `bert-base-uncased` (which could be loaded in huggingface) to preload the encoder of our model.
#### Example: training on Red-Multi
```console
python /path/train_TranExtrac.py --train_data_path /path/train_Red-Mulit.csv --val_data_path /path/val_Red-Multi.csv --pretrained_model_path /path/bert-base-uncased  --model_save_path /path/model_save_path --num_train_epochs 15 --batch_size 48 --warmup_steps 8000 --eval_steps 400 --learning_rate 2e-5
```

#### Example: training on CPRM

```console
python /path/train_TranExtrac.py --train_data_path /path/train_CPRM.csv --val_data_path /path/val_CPRM.csv --pretrained_model_path /path/bert-base-uncased  --model_save_path /path/model_save_path --num_train_epochs 15 --batch_size 48 --warmup_steps 16000 --eval_steps 2000 --learning_rate 2e-5
```

#### 

### Evaluate

After completing the training process, the best weight of model named `best_extract_model.pt` will be stored in the folder to save checkpoints.  You can run the following command to get the Rouge score on test set

```console
python /path/evaluate_new.py --test_data_path /path/test_data_path --pretrained_model_path /path/pretrained_bert_model --model_finetuned_path /path/finetuned_best_model  --num_beams 4 --out_sentence_positions 0
```

#### Example: evaluating on Red-Multi

```console
python /path/evaluate_new.py --test_data_path /path/test_Red-Mulit.csv --pretrained_model_path /path/bert-base-uncased --model_finetuned_path /path/best_extract_model.pt  --num_beams 4 --out_sentence_positions 0
```

#### Example: evaluating on CPRM

```console
python /path/evaluate_new.py --test_data_path /path/test_CPRM.csv --pretrained_model_path /path/bert-base-uncased --model_finetuned_path /path/best_extract_model.pt  --num_beams 4 --out_sentence_positions 0
```



