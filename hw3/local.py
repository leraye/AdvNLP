#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:47:10 2019

@author: heqingye
"""

from hmmpar import SmoothedMLE
from data import Corpus
import torch

path = "31210-s19-hw3/"
corpus = Corpus(path)

class NaiveDecoder(object):
    def __init__(self, MLE):
        self.par = MLE
        self.emitter(MLE)
        
    def emitter(self, MLE):
        v, idx = torch.topk(MLE.emission_matrix, 1, 0)
        self.most_prob_tag = idx[0,]
        self.most_prob = v[0,]
                        
    def decode(self, data):
        total_log = 0.
        decoded = []
        for s, t in zip(data[0], data[1]):
            x = torch.ones(len(s), dtype=torch.long)
            prev = 0
            for i in range(len(s)):
                w = s[i].item()
                wt = self.most_prob_tag[w]
                x[i] = wt
                total_log += self.most_prob[w].item() + self.par.transition_prob(prev, wt)
                prev = wt
            decoded.append(x)
            total_log += self.par.transition_prob(prev, 1)
        total_correct = 0.
        total_len = 0.
        for t, y in zip(data[1], decoded):
            total_correct += t[1:-1].eq(y).sum().item()
            total_len += len(y)
        return total_log, total_correct * 100. / total_len
    
MLE = SmoothedMLE(corpus)
    
NDecoder = NaiveDecoder(MLE)
x, y = NDecoder.decode(corpus.valid)
print(x)
print(y)