from transformers import BertTokenizer, BertModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.modeling_outputs import BaseModelOutput
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from typing import List
import math
from train_TranExtrac import DataCollatorForTranExtrac

import torch
import torch.nn as nn

from neural import MultiHeadedAttention, PositionwiseFeedForward

cuda0 = torch.device('cuda:0')
cpu = torch.device('cpu')

class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

class Bert(nn.Module):
    def __init__(self, pretrained_model_path):
        super(Bert, self).__init__()

        self.model = BertModel.from_pretrained(pretrained_model_path)

    def forward(self, x, segs, mask):
        output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
        top_vec = output.last_hidden_state
        return top_vec


class BertSum(nn.Module):
    def __init__(self, args):
        super(BertSum, self).__init__()
        self.args = args
        pretrained_model_path = args.pretrained_model_path


        self.bert = Bert(pretrained_model_path=pretrained_model_path)
        max_position_input = args.max_position_input
        if(max_position_input>512):
            my_pos_embeddings = nn.Embedding(args.max_position_input, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


    def forward(self, src, segs, clss, mask_src, mask_cls, labels=None):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        return BaseModelOutput(last_hidden_state=sents_vec)



class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_token_id

    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __call__(self, examples: List[dict]):
        input_ids = [example['input_ids'] for example in examples]
        token_type_ids = [example['token_type_ids'] for example in examples]
        cls = [example['cls'] for example in examples]
        labels = [example['labels'] for example in examples]

        new_labels = []
        for i, label in enumerate(labels):
            new_label = [0] * len(cls[i])
            for j in label:
                new_label[j] = 1
            new_labels.append(new_label)
        labels = new_labels


        input_ids = torch.tensor(self._pad(input_ids, 0))

        token_type_ids = torch.tensor(self._pad(token_type_ids, 0))
        attention_mask = 1 - (token_type_ids == 0).int()

        cls = torch.tensor(self._pad(cls, -1))
        labels = torch.tensor(self._pad(labels, 0))
        mask_cls = 1 - (cls == -1).int()
        cls[cls == -1] = 0


        output_dict = {}

        output_dict['src'] = input_ids
        output_dict['segs'] = token_type_ids
        output_dict['clss'] = cls
        output_dict['mask_src'] = attention_mask
        output_dict['mask_cls'] = mask_cls
        output_dict['labels'] = labels


        return output_dict


class CustomTrainer(Trainer):
    def __init__(
        self,
        log_steps=4,
        **kwargs
    ):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get('labels')
        mask_cls = inputs.get('mask_cls')

        outputs_model = model(**inputs)
        outputs = outputs_model.last_hidden_state


        loss_fct = nn.BCELoss(reduction='none')
        loss = loss_fct(outputs, labels.float())
        loss = loss * mask_cls.float()
        loss = loss.sum() / loss.size(0)


        return (loss, outputs_model) if return_outputs else loss