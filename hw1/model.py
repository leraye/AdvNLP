# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, device, r=None, f=None, p=None):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, batch_first = True)
        self.decoder = nn.Embedding(ntoken, nhid, padding_idx=0)

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.nsamples = r
        self.power = f
        self.ntoken = ntoken
        self.div = device
        self.dic = torch.LongTensor(list(range(self.ntoken))).to(self.div)
        if p is not None:
            self.prob = torch.pow(p, self.power)
            self.prob[0] = 0
        else:
            self.prob = p
            
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.weight.data[0,:] = 0
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data[0,:] = 0

    def forward(self, input, target, hidden):
        emb = self.encoder(input)
        output, _ = self.rnn(emb, hidden) # bsz x seq x nhid
        
        if self.prob is not None:
            op = output.contiguous().view(-1, self.nhid) # bsz * seq x nhid
            de = self.decoder(target).contiguous().view(-1, self.nhid) # bsz * seq x nhid
            prob = self.prob.expand((output.size(0) * output.size(1), self.ntoken))
            nwords = torch.multinomial(prob, self.nsamples, True).to(self.div)
            ng_emb = self.decoder(nwords) # bsz * seq x nsamples x nhid
            decoded = None
            sp = torch.bmm(de.unsqueeze(1), op.unsqueeze(2)).squeeze(2)
            sn = torch.bmm(ng_emb, op.unsqueeze(2)).squeeze()
        else:
            w = self.decoder(self.dic).t().contiguous() # nhid x ntoken
            decoded = torch.matmul(output, w)
            sp = None
            sn = None
            
        return decoded, sp, sn
    
    def evaluate(self, input, hidden):
        emb = self.encoder(input)
        output, _ = self.rnn(emb, hidden)
        w = self.decoder(self.dic).t().contiguous() # nhid x ntoken
        decoded = F.logsigmoid(torch.matmul(output, w))
        return decoded
        

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
