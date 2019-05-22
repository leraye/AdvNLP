#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:00:33 2019

@author: heqingye
"""

import torch
import numpy as np
import os
import argparse

class ViterbiDecoder(object):
    def __init__(self, path, total_tags):
        self.ntags = total_tags
        assert os.path.exists(path)
        self.TMatrix = torch.load(path + 'tmatrix.pt')
        self.EMatrix = torch.load(path + 'ematrix.pt')
        
    def exactSearch(self, s):
        table = [{}]
        bp = []
        for i in range(2, self.ntags):
            table[0][i] = 0
        N = len(s)
        for i in range(N):
            V = table.pop()
            table.append({})
            bp.append({})
            w = s[i].item()
            for j in range(2, self.ntags):
                if i == 0:
                    table[-1][j] = self.EMatrix[j, w].item() + self.TMatrix[0, j].item() + V[j]
                else:
                    table[-1][j] = self.EMatrix[j, w].item()
                    best = -np.Inf
                    for k in range(2, self.ntags):
                        p = self.TMatrix[k, j].item() + V[k]
                        if p > best:
                            best = p
                            bp[-1][j] = k
                    table[-1][j] += best
        return table, bp
            
    def decode(self, data):
        total_log = 0.
        decoded = []
        for s, _ in zip(data[0], data[1]):
            best = -np.Inf
            table, bp = self.exactSearch(s)
            x = torch.LongTensor(len(s))
            V = table.pop()
            for j in range(2, self.ntags):
                if V[j] + self.TMatrix[j, 1].item() > best:
                    best = V[j] + self.TMatrix[j, 1].item()
                    x[-1] = j
            i = len(s) - 1
            total_log += best
            while len(bp) > 0:
                v = bp.pop()
                if len(v) > 0:
                    x[i - 1] = v[x[i].item()]
                i -= 1
            decoded.append(x)
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
VDecoder = ViterbiDecoder(args.par, ntags)
valid = torch.load(args.data)
x, y = VDecoder.decode(valid)
print(x)
print(y)