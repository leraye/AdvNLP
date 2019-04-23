#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 00:08:48 2019

@author: heqingye
"""

import torch.nn as nn


class SimpleRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers):
        super(SimpleRNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, batch_first = True)
        self.decoder = nn.Linear(nhid, ntoken, False)

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
            
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hid = self.rnn(emb, hidden) # bsz x seq x nhid
        decoded = self.decoder(output.contiguous().view(output.size(0) * output.size(1), -1))
            
        return decoded, hid
        

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))