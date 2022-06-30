import torch
import torch.nn as nn
import torchvision.models as models
import sys

from lm import WordEmbedding, QuestionEmbedding
from remix import nn_reasoning_block

class BaseModel(nn.Module):
    def __init__(self, nn_w_emb, nn_q_emb, nn_reasoning, args):
        super(BaseModel, self).__init__()
        self.nn_w_emb = nn_w_emb
        self.nn_q_emb = nn_q_emb
        self.nn_reasoning = nn_reasoning
        
    def forward(self, img_cf, q, qlens):
        """Forward"""
        w_emb = self.nn_w_emb(q)
        q_emb = self.nn_q_emb.forward_all(w_emb)
        out = self.nn_reasoning(q_emb, img_cf, qlens)
        return out


def build_block(dataset, args):
    m_len = 14
    nn_w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.5)
    nn_q_emb = QuestionEmbedding(300, args.num_hid, 1, False, 0.0)
    nn_reasoning = nn_reasoning_block(m_len, args.num_hid, 36, 2048, args.num_hid, dataset.num_ans_candidates)
    return BaseModel(nn_w_emb, nn_q_emb, nn_reasoning, args)
