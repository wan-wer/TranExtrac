import datasets
import random
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import GenerationConfig
from modeling_TranExtrac import *
import pyrouge
import shutil
import time

import wandb
import numpy as np
import os
import torch
import numpy as np
import random
import argparse


from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def wandb_set(wandb_project='t5_ext', wandb_log_model='true', run_name='no_ext_max_4096_epoch_10_wordSplit', report_to='wandb'):
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ['WANDB_LOG_MODEL'] = wandb_log_model
    wandb.login(key='') # to add
    wandb.init(project=wandb_project)
    run_name = run_name
    wandb.run.name = run_name
    report_to = report_to

def restore_data(data_df):
    if 'text_split' in data_df.columns:
        data_df['text_split'] = data_df['text_split'].parallel_apply(lambda x: eval(x))
    data_df['input_ids'] = data_df['input_ids'].parallel_apply(lambda x: eval(x))
    data_df['token_type_ids'] = data_df['token_type_ids'].parallel_apply(lambda x: eval(x))
    data_df['cls'] = data_df['cls'].parallel_apply(lambda x: eval(x))
    data_df['labels'] = data_df['labels'].parallel_apply(lambda x: eval(x))
    return data_df

val_dataset = None
score_max = float('-inf')

@dataclass
class DataCollatorForTranExtrac:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

        def pad_helper(l, pad_num, pad_id=0):
            return l + [pad_id] * pad_num

        def pad_fun(column_series, pad_id=0):
            max_len = max([len(x) for x in column_series])
            return [pad_helper(x, max_len - len(x), pad_id=pad_id) for x in column_series]
        items_need_pad = ['input_ids', 'token_type_ids', 'cls', 'labels']
        output_dict = {}

        output2pad = pd.DataFrame(features)
        output2pad = output2pad.to_dict(orient = 'list')
        labels = output2pad['labels']
        for i, label in enumerate(labels):
            output2pad['labels'][i] = (np.array(label) + 2).tolist() + [0]
        for col in items_need_pad:
            if col == 'cls':
                output_dict[col] = pad_fun(output2pad[col], pad_id=-1)
            elif col == 'labels':
                output_dict[col] = pad_fun(output2pad[col], pad_id=-100)
            elif col == 'token_type_ids':
                output_dict[col] = pad_fun(output2pad[col], pad_id=0)
            else:
                output_dict[col] = pad_fun(output2pad[col], pad_id=self.tokenizer.pad_token_id)


        features = self.tokenizer.pad(
            output_dict,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        input_ids = features['input_ids']
        attention_mask = 1 - (input_ids == 0).int()
        cls = features['cls']
        mask_cls = 1 - (cls == -1).int()
        cls[cls == -1] = 0

        features['attention_mask'] = attention_mask
        features['mask_cls'] = mask_cls
        features['cls'] = cls

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

score_max = float('-inf')
val_dataset = None
def give_rouge_score(ele):
    from rouge import Rouge
    rouge = Rouge()
    hyp = ele['hyp']
    ref = ele['ref']
    scores = rouge.get_scores(hyp, ref, avg=True)
    return scores
def train(args):

    seed =args.seed

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    eval_steps = args.eval_steps


    model_save_path = args.model_save_path
    pretrained_model_path = args.pretrained_model_path
    load_finetuned_model_path = args.load_finetuned_model_path

    report_to = args.report_to
    num_train_epochs = args.num_train_epochs
    start_wandb = args.start_wandb
    if not start_wandb:
        report_to = 'none'
    batch_size = args.batch_size

    label_smoothing = args.label_smoothing
    log_steps = args.log_steps
    warmup_steps = args.warmup_steps
    learning_rate = args.learning_rate
    gradient_accumulation_steps = args.gradient_accumulation_steps
    result_can_path = args.result_can_path
    result_gold_path = args.result_gold_path
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    print('finish loading dataset')

    train_data = restore_data(train_data)
    train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_data = restore_data(val_data)
    
    print('finish restoring dataset')

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)

    train_dataset = Dataset.from_pandas(train_data)
    global val_dataset
    val_dataset = Dataset.from_pandas(val_data)

    print('finish converting dataframe to dataset')

    ext_ff_size = args.ext_ff_size
    ext_heads = args.ext_heads
    ext_dropout = args.ext_dropout
    ext_layers = args.ext_layers
    max_position_input = args.max_position_input
    param_init = args.param_init
    param_init_glorot = args.param_init_glorot

    config = TranExtracConfig(pretrained_model_path=pretrained_model_path, ext_ff_size=ext_ff_size, ext_heads=ext_heads, ext_dropout=ext_dropout,
                              ext_layers=ext_layers, max_position_input=max_position_input, param_init=param_init, param_init_glorot=param_init_glorot,)
    model = TranExtracForConditionalGeneration(config)
    if load_finetuned_model_path != '':
        state_dict = torch.load(load_finetuned_model_path)
        print('loading finetuned model')
        model.load_state_dict(state_dict)
        print('finish loading finetuned model')

    args = Seq2SeqTrainingArguments(
        model_save_path,
        evaluation_strategy="steps",
        save_strategy='steps',
        eval_steps=eval_steps,
        save_steps=eval_steps,
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        generation_max_length=15,
        logging_steps=log_steps,
        report_to=report_to,
        remove_unused_columns=False,
        warmup_steps=warmup_steps,

    )
    data_collator = DataCollatorForTranExtrac(tokenizer, model=model)

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        predictions = np.where(predictions != -100, predictions, 0)
        labels = np.where(labels != -100, labels, 0)

        pred_str = []
        summary_standard_str = []
        predictions = predictions - 2
        for i, item in enumerate(predictions.tolist()):

            item_available = [v for v in item if v >= 0]
            item_available = sorted(item_available)

            text_split = val_data.loc[i, 'text_split']
            summary_standard = val_data.loc[i, 'summary']
            text_extract = [text_split[v] for v in item_available if v < len(text_split)]
            text_extract = ' '.join(text_extract)

            if text_extract == '':
                continue
            pred_str.append(text_extract)
            summary_standard_str.append(summary_standard)

        data_df = pd.DataFrame({'hyp': pred_str, 'ref': summary_standard_str})
        scores = data_df.parallel_apply(give_rouge_score, axis=1)

        rouge1 = scores.apply(lambda x: x['rouge-1']['f'])
        rouge2 = scores.apply(lambda x: x['rouge-2']['f'])
        rougel = scores.apply(lambda x: x['rouge-l']['f'])
        rouge1 = rouge1.mean()
        rouge2 = rouge2.mean()
        rougeL = rougel.mean()

        score_rouge = rouge1 * 0.3 + rouge2 * 0.3 + rougeL * 0.4
        global score_max
        if score_rouge > score_max:
            path = os.path.join(model_save_path, 'best_extract_model.pt')
            torch.save(model.state_dict(), path)
            score_max = score_rouge
        return {'rouge-1': rouge1, 'rouge-2': rouge2, 'rouge-L': rougeL}


    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()




if __name__ == '__main__':
    parse = argparse.ArgumentParser() 

    seed = 34
    setup_seed(seed)

    parse.add_argument('--seed', type=int, default=34, required=False)

    parse.add_argument('--start_wandb', type=int, default=0, required=False, help='0: False, 1: True')
    parse.add_argument('--wandb_project', type=str, default='Extract_multinews', required=False)
    parse.add_argument('--wandb_log_model', type=str, default='true', required=False)
    parse.add_argument('--run_name', type=str, default='TranExtrac', required=False)
    parse.add_argument('--report_to', type=str, default='wandb', required=False)

    parse.add_argument('--num_train_epochs', type=int, default=15, required=False)
    parse.add_argument('--batch_size', type=int, default=48, required=False)
    parse.add_argument('--label_smoothing', type=float, default=0.1, required=False)
    parse.add_argument('--log_steps', type=int, default=32, required=False)
    parse.add_argument('--warmup_steps', type=int, default=4000, required=False)
    parse.add_argument('--eval_steps', type=int, default=200, required=False)
    parse.add_argument('--learning_rate', type=float, default=2e-5, required=False)
    parse.add_argument('--gradient_accumulation_steps', type=int, default=2, required=False)

    parse.add_argument('--train_data_path', type=str,
                       default='/path/train_multinews_new_label.csv', required=False)
    parse.add_argument('--val_data_path', type=str,
                       default='/path/val_multinews_new_label.csv', required=False)

    parse.add_argument('--pretrained_model_path', type=str,
                       default='/path/bert-base-uncased', required=False)
    parse.add_argument('--model_save_path', type=str,
                       default='/path/model_save_path', required=False)
    parse.add_argument('--load_finetuned_model_path', type=str,
                       default='', required=False)
    parse.add_argument('--result_can_path', type=str, default='./val.candidate', required=False)
    parse.add_argument('--result_gold_path', type=str, default='./val.gold', required=False)

    parse.add_argument('--ext_ff_size', type=int, default=2048, required=False)
    parse.add_argument('--ext_heads', type=int, default=8, required=False)
    parse.add_argument('--ext_dropout', type=float, default=0.2, required=False)
    parse.add_argument('--ext_layers', type=int, default=2, required=False)
    parse.add_argument('--max_position_input', type=int, default=512, required=False)


    parse.add_argument("--param_init", default=0, type=float)
    parse.add_argument("--param_init_glorot", type=bool,default=True)

    args = parse.parse_args()  
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    start_wandb = args.start_wandb
    wandb_project = args.wandb_project
    wandb_log_model = args.wandb_log_model
    run_name = args.run_name
    report_to = args.report_to


    if start_wandb:
        os.environ['WANDB_DIR'] = os.getcwd() + "/wandb/"
        os.environ['WANDB_CACHE_DIR'] = os.getcwd() + "/wandb/.cache/"
        os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + "/wandb/.config/"
        wandb_set(wandb_project=wandb_project, wandb_log_model=wandb_log_model, run_name=run_name,report_to=report_to)
    print('start train function')
    train(args)
