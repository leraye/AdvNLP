#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:53:28 2019

@author: heqingye
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nlayers, nclass):
        super(SimpleClassifier, self).__init__()
        self.encoder = nn.Embedding(ntoken + 1, ninp, ntoken)
        self.decoder = nn.Linear(ninp, nclass, False)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, mask):
        emb = self.encoder(input) * mask.unsqueeze(2)
        lens = torch.sum(mask, 1, True)
        avg = emb.sum(1) / lens
        decoded = self.decoder(avg)
        return decoded
    
class CosAttenClassifier(SimpleClassifier):
    
    def __init__(self, ntoken, ninp, nlayers, nclass):
        super(CosAttenClassifier, self).__init__(ntoken, ninp, nlayers, nclass)
        self.u = nn.Parameter(torch.randn(ninp))
        
    def forward(self, input, mask):
        emb = self.encoder(input) * mask.unsqueeze(2)
        cossim = torch.exp(F.cosine_similarity(self.u.unsqueeze(0).unsqueeze(1).expand_as(emb), emb, 2)) * mask # bsz x seq
        attn = cossim / torch.sum(cossim, 1, True)
        output = (emb * attn.unsqueeze(2)).sum(1)
        decoded = self.decoder(output)
        return decoded
        
class SelfAttenClassifier(SimpleClassifier):
    
    def __init__(self, ntoken, ninp, nlayers, nclass, resid=True):
        super(SelfAttenClassifier, self).__init__(ntoken, ninp, nlayers, nclass)
        self.residual = resid
        
    def forward(self, input, mask):
        emb = self.encoder(input) * mask.unsqueeze(2) # bsz x seq x nhid
        x = torch.bmm(emb, emb.transpose(1, 2)) # bsz x seq x seq
        y = torch.exp(x.sum(2)) * mask
        attn = y / torch.sum(y, 1, True) # bsz x seq
        output = (emb * attn.unsqueeze(2)).sum(1)
        if self.residual:
            lens = torch.sum(mask, 1, True)
            avg = emb.sum(1) / lens
            out = output + avg
            decoded = self.decoder(out)
        else:
            decoded = self.decoder(output)
        return decoded
    
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
        x = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(self.d)
        y = torch.exp(x.sum(2)) * mask
        attn = y / torch.sum(y, 1, True)
        output = (V * attn.unsqueeze(2)).sum(1)
        lens = torch.sum(mask, 1, True)
        avg = V.sum(1) / lens
        out = self.norm(self.drop(output) + avg)
        decoded = self.decoder(out)
        return decoded
    
class BiLSTMAttention(nn.Module):
    
    def __init__(self, ntoken, ninp, nlayers, nclass):
        super(BiLSTMAttention, self).__init__()
        self.encoder = nn.Embedding(ntoken + 1, ninp, ntoken)
        self.rnn = nn.LSTM(ninp, ninp, batch_first=True, bidirectional=True)
        self.U = nn.Linear(ninp * 2, ninp)
        self.v = nn.Parameter(torch.randn(ninp))
        stdv = 1. / math.sqrt(ninp) # initialization turns out to be very important
        self.v.data.normal_(mean=0, std=stdv)
        self.decoder = nn.Linear(ninp, nclass, False)
        
        self.init_weights()
        self.nlayers = nlayers
        self.nhid = ninp
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, input, mask, hidden=None):
        emb = self.encoder(input) * mask.unsqueeze(2)
        if hidden is None:
            hidden = self.init_hidden(input.size(0))
        output, _ = self.rnn(emb, hidden) # bsz x seq_len x 2*hsz
        energy = torch.tanh(self.U(output)) # bsz x seq_len x hsz
        att = torch.matmul(energy, self.v.unsqueeze(1)).squeeze() # bsz x seq_len x 1
        y = torch.exp(att) * mask
        scores = y / y.sum(1, True)
        out = (emb * scores.unsqueeze(2)).sum(1)
        decoded = self.decoder(out)
        return decoded
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers * 2, bsz, self.nhid),
                weight.new_zeros(self.nlayers * 2, bsz, self.nhid))