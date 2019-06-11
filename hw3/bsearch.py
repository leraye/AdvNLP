#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:59:17 2019

@author: heqingye
"""

import torch
import heapq
import operator
import os
import argparse

class BeamSearchDecoder(object):
    def __init__(self, path, b, total_tags):
        assert b > 0
        self.ntags = total_tags
        self.bsize = b
        assert os.path.exists(path)
        self.TMatrix = torch.load(path + 'tmatrix.pt')
        self.EMatrix = torch.load(path + 'ematrix.pt')
    
    def bsearch(self, s):
        prev = [[0]]
        path = []
        scores = torch.zeros(self.ntags)
        S = len(s)
        for i in range(S):
            scores = scores + self.TMatrix[prev[-1],] + self.EMatrix[:,s[i].item()]
            if i == S - 1:
                scores = scores + self.TMatrix[:,1]
            seen = {}
            edges = {}
            K = len(prev[-1])
            for k in range(K):
                for j in range(2, self.ntags):
                    if j not in seen or scores[k, j] > seen[j]:
                        edges[j] = k
                        seen[j] = scores[k, j].item()
            if i < S - 1:
                beam = heapq.nlargest(self.bsize, seen.items(), key=operator.itemgetter(1))
            else:
                beam = heapq.nlargest(1, seen.items(), key=operator.itemgetter(1))
            prev.append([x[0] for x in beam])
            path.append([edges[x[0]] for x in beam])
            scores = torch.Tensor([x[1] for x in beam]).unsqueeze(1).expand(len(beam), self.ntags)
            
        label = torch.ones(S, dtype=torch.long)
        pt = 0
        j = S - 1
        while len(path) > 0:
            label[j] = prev[-1][pt]
            pt = path[-1][pt]
            path.pop()
            prev.pop()
            j -= 1
        return beam[0][1], label
    
    def decode(self, data):
        total_log = 0.
        decoded = []
        for s, _ in zip(data[0], data[1]):
            scr, t = self.bsearch(s)
            total_log += scr
            decoded.append(t)
        total_correct = 0.
        total_len = 0.
        for t, y in zip(data[1], decoded):
            total_correct += t[1:-1].eq(y).sum().item()
            total_len += len(y)
        return total_log, total_correct * 100. / total_len

parser = argparse.ArgumentParser(description='Beam Search')
parser.add_argument('--par', type=str, default="./",
                    help='location of the parameter matrix')
parser.add_argument('--data', type=str, default="./valid.pt",
                    help='location of the data matrix')
parser.add_argument('--bsize', type=int, default=5,
                    help='beam size')
args = parser.parse_args()
beam_size = args.bsize
ntags = 52
BDecoder = BeamSearchDecoder(args.par, beam_size, ntags)
valid = torch.load(args.data)
x, y = BDecoder.decode(valid)
print(x)
print(y)