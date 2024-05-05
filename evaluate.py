from modeling_TranExtrac import *
from datasets import Dataset
from train_TranExtrac import DataCollatorForTranExtrac
from transformers import BertTokenizer
import argparse
import os
import json
import pyrouge
import shutil
import time
import os
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
import numpy as np

def restore_data(data_df):

    data_df['text_split'] = data_df['text_split'].apply(lambda x: eval(x))
    data_df['input_ids'] = data_df['input_ids'].apply(lambda x: eval(x))
    data_df['token_type_ids'] = data_df['token_type_ids'].apply(lambda x: eval(x))
    data_df['cls'] = data_df['cls'].apply(lambda x: eval(x))
    data_df['labels'] = data_df['labels'].apply(lambda x: eval(x))

    return data_df
def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set
def _block_tri(c, p):
    tri_c = _get_ngrams(3, c.split())
    for s in p:
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s)) > 0:
            return True
    return False
def give_summary_extract(list_model_out, test_data):
    pred_str = []
    for i in test_data.index:
        pred = list_model_out[i]
        pred = [_ - 2 for _ in pred if _ - 2 >= 0]
        text_split = test_data.loc[i, 'text_split']

        new_pred = []
        for q,x in enumerate(pred):
            if x not in new_pred:
                new_pred.append(x)
        pred = new_pred

        index_sort = sorted(pred)

        text_extract = [text_split[q] for q in index_sort]
        text_extract = ' '.join(text_extract)

        pred_str.append(text_extract)
    return pred_str


score_max = float('-inf')


@dataclass
class MyDataCollatorForTranExtrac:
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
        # mask_tgt = 1 - (tgt == 0)
        cls = features['cls']
        mask_cls = 1 - (cls == -1).int()
        cls[cls == -1] = 0

        features['attention_mask'] = attention_mask
        features['mask_cls'] = mask_cls
        features['cls'] = cls
        return features

def get_trainer(model, pretrained_model_path, batch_size=1):
    training_args = Seq2SeqTrainingArguments(output_dir='result', warmup_steps=150, evaluation_strategy=IntervalStrategy.STEPS,
                                      eval_steps=150,
                                      load_best_model_at_end=True, save_total_limit=1, num_train_epochs=50,
                                      save_steps=150,
                                      logging_steps=1, per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      remove_unused_columns=False)
    tokenizer = BertTokenizer(vocab_file=os.path.join(pretrained_model_path, 'vocab.txt'))
    data_collator = MyDataCollatorForTranExtrac(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    return trainer
def evaluate(args):

    test_data_path = args.test_data_path
    model_finetuned_path = args.model_finetuned_path
    pretrained_model_path = args.pretrained_model_path
    num_beams = args.num_beams
    out_sentence_positions = args.out_sentence_positions
    sentence_positions_path = args.sentence_positions_path
    test_data = pd.read_csv(test_data_path)
    test_data = restore_data(test_data)


    test_dataset = Dataset.from_pandas(test_data)

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

    state_dict = torch.load(model_finetuned_path)
    print('loading model')
    model.load_state_dict(state_dict)
    print('finish loading model')

    model.eval()
    trainer = get_trainer(model, pretrained_model_path, batch_size=1)
    dataloader = trainer.get_test_dataloader(test_dataset)
    model.eval()
    output = torch.Tensor()
    labels = torch.Tensor()

    list_model_out = []
    print('num_beams: {}'.format(num_beams))
    with torch.no_grad():
        for index,item in enumerate(tqdm(dataloader)):
            for key, value in item.items():
                item[key] = value.to(next(model.parameters()).device)
            out = model.generate(**item, do_sample=False, min_new_tokens=1, max_new_tokens=30, num_beams=num_beams, num_beam_groups=1, length_penalty=1.0)
            out_list = out.tolist()
            list_model_out.extend(out_list)

    if out_sentence_positions:
        position_list = []

        for i in test_data.index:

            pred = list_model_out[i]
            pred = [_ - 2 for _ in pred if _ - 2 >= 0]
            new_pred = []
            for q,x in enumerate(pred):
                if x not in new_pred:
                    new_pred.append(x)
            position_list.append(sorted(new_pred))
        
        position_json = json.dumps(position_list)
        with open(sentence_positions_path, 'w', encoding='utf-8') as ff:
            ff.write(position_json)

    predict_summary = give_summary_extract(list_model_out, test_data)

    can_path = args.result_can_path
    gold_path = args.result_gold_path
    pred_str = []
    summary_standard_str = []
    with open(can_path, 'w') as save_pred:
        with open(gold_path, 'w') as save_gold:
            for i in test_data.index:
                summary_standard = test_data.loc[i, 'summary'].strip()
                _pred = predict_summary[i].strip()
                pred_str.append(_pred)
                summary_standard_str.append(summary_standard)
            for i in range(len(summary_standard_str)):
                save_gold.write(summary_standard_str[i].strip() + '\n')
            for i in range(len(pred_str)):
                save_pred.write(pred_str[i].strip() + '\n')
    
    os.system('files2rouge {} {}'.format(can_path, gold_path))







if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_data_path', type=str,
                       default='/path/test_Red-Multi.csv', required=False)
    parse.add_argument('--pretrained_model_path', type=str,
                       default='/path/bert-base-uncased', required=False)
    parse.add_argument('--model_finetuned_path', type=str,
                       default='/path/best_extract_model.pt', required=False)


    parse.add_argument('--ext_ff_size', type=int, default=2048, required=False)
    parse.add_argument('--ext_heads', type=int, default=8, required=False)
    parse.add_argument('--ext_dropout', type=float, default=0.2, required=False)
    parse.add_argument('--ext_layers', type=int, default=2, required=False)
    parse.add_argument('--max_position_input', type=int, default=512, required=False)


    parse.add_argument("--param_init", default=0, type=float)
    parse.add_argument("--param_init_glorot", type=bool,default=True)

    parse.add_argument('--result_can_path', type=str, default='./val.candidate', required=False)
    parse.add_argument('--result_gold_path', type=str, default='./val.gold', required=False)
    
    parse.add_argument('--num_beams', type=int, default=4, required=False)   

    parse.add_argument('--out_sentence_positions', type=int, default=1, required=False) 
    parse.add_argument('--sentence_positions_path', type=str, default='./sentence_positions.txt', required=False)

    args = parse.parse_args()  
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    result = evaluate(args)



