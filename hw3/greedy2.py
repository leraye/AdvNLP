#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:19:34 2019

@author: heqingye
"""

import torch
import numpy as np
import os
import argparse

class GreedyRLDecoder(object):
    def __init__(self, path, total_tags):
        self.ntags = total_tags
        assert os.path.exists(path)
        self.TMatrix = torch.load(path + 'tmatrix.pt')
        self.EMatrix = torch.load(path + 'ematrix.pt')
        
    def greedySearch(self, s):
        N = len(s)
        x = torch.ones(N, dtype=torch.long)
        nxt = 1
        total = 0.
        for i in range(N - 1, -1, -1):
            w = s[i].item()
            best = -np.Inf
            for j in range(2, self.ntags):
                p = self.EMatrix[j, w].item() + self.TMatrix[j, nxt].item()
                if i == 0:
                    p += self.TMatrix[0, j].item()
                if p > best:
                    best = p
                    x[i] = j
            nxt = x[i]
            total += best
        return x, total
    
    def decode(self, data):
        total_log = 0.
        decoded = []
        for s, _ in zip(data[0], data[1]):
            x, p = self.greedySearch(s)
            decoded.append(x)
            total_log += p
        total_correct = 0.
        total_len = 0.
        for t, y in zip(data[1], decoded):
            total_correct += t[1:-1].eq(y).sum().item()
            total_len += len(y)
        return total_log, total_correct * 100. / total_len
    
parser = argparse.ArgumentParser(description='Viterbi Exact Search')
parser.add_argument('--par', type=str, default="./",
                    help='location of the parameter matrix')
parser.add_argument('--data', type=str, default="./valid.pt",
                    help='location of the data matrix')

args = parser.parse_args()
ntags = 52
    
GRLDecoder = GreedyRLDecoder(args.par, ntags)
valid = torch.load(args.data)
x, y = GRLDecoder.decode(valid)
print(x)
print(y)