#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:15:43 2019

@author: heqingye
"""

import torch
import math
import copy
import torch.nn as nn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
#        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(2000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        y = self.pe[:, :x.size(1)]
        return x + y
#        return self.dropout(x + y)

class ScaledDPAttention(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nlayers, nclass, dropout=0.5):
        super(ScaledDPAttention, self).__init__()
        self.encoderv = nn.Embedding(ntoken + 1, ninp, ntoken)
        self.encoderq = nn.Embedding(ntoken + 1, ninp, ntoken)
        self.encoderk = nn.Embedding(ntoken + 1, ninp, ntoken)
        self.decoder = nn.Linear(ninp, nclass, False)
        
        self.init_weights()
        
        self.d = ninp
        self.drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(self.d, elementwise_affine=False)
        self.pe = PositionalEncoding(self.d, 0.1)

    def init_weights(self):
        initrange = 0.1
        self.encoderv.weight.data.uniform_(-initrange, initrange)
        self.encoderq.weight.data.uniform_(-initrange, initrange)
        self.encoderk.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, mask):
        V = self.encoderv(input) * mask.unsqueeze(2)
        K = self.encoderk(input) * mask.unsqueeze(2)
        Q = self.encoderq(input) * mask.unsqueeze(2)
        value = self.pe(V)
        x = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(self.d)
        y = torch.exp(x.sum(2)) * mask
        attn = y / torch.sum(y, 1, True)
        output = (value * attn.unsqueeze(2)).sum(1)
#        output = (V * attn.unsqueeze(2)).sum(1)
        lens = torch.sum(mask, 1, True)
        avg = value.sum(1) / lens
#        avg = V.sum(1) / lens
        out = self.norm(self.drop(output) + avg)
        decoded = self.decoder(out)
        return decoded

    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, ntoken, nhid, nlayers, nheads, nclass, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        assert nhid % nheads == 0
        self.encoderv = nn.Embedding(ntoken + 1, nhid, ntoken)
        self.encoderq = nn.Embedding(ntoken + 1, nhid, ntoken)
        self.encoderk = nn.Embedding(ntoken + 1, nhid, ntoken)
        self.linears = clones(nn.Linear(nhid, nhid), 4)
        self.h = nheads
        self.d = nhid
        self.decoder = nn.Linear(nhid, nclass, False)
        
        self.init_weights()
        
        self.d_k = nhid // self.h
        self.drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(self.d, elementwise_affine=False)
        
    def init_weights(self):
        initrange = 0.1
        self.encoderv.weight.data.uniform_(-initrange, initrange)
        self.encoderq.weight.data.uniform_(-initrange, initrange)
        self.encoderk.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, input, mask):
        V = self.encoderv(input) * mask.unsqueeze(2)
        K = self.encoderk(input) * mask.unsqueeze(2)
        Q = self.encoderq(input) * mask.unsqueeze(2)
        bsz = input.size(0)
        query, key, value = [l(x).view(bsz, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (Q, K, V))]
        x = torch.matmul(query, key.transpose(-2, -1))
        y = torch.exp(x.sum(2)) * mask.unsqueeze(1) # bsz x h x seq
        attn = y / torch.sum(y, -1, True)
        output = (value * attn.unsqueeze(-1)).sum(-2) # bsz x h x d_k
        out = output.transpose(1, 2).contiguous().view(bsz, self.h * self.d_k)
        lens = torch.sum(mask, 1, True)
        avg = V.sum(1) / lens
        ded = self.norm(self.drop(self.linears[-1](out)) + avg)
        decoded = self.decoder(ded)
        return decoded