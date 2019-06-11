#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:45:37 2019

@author: heqingye
"""

import torch
import os
import argparse

class GibbsSampler(object):
    def __init__(self, path, K, p, total_tags=52, step=0, MBR=False, burn_in=False):
        assert K > 0
        self.niter = K
        self.ntags = total_tags
        self.step = step
        self.pow = p
        self.mbr = MBR
        if MBR:
            self.bi = burn_in
        assert os.path.exists(path)
        self.TMatrix = torch.load(path + 'tmatrix.pt')
        self.EMatrix = torch.load(path + 'ematrix.pt')
        
    def sampler(self, s):
        S = len(s)
        v = torch.randint(2, self.ntags, (S,))
        if self.mbr:
            samples = torch.zeros(self.ntags, S)
        p = self.pow
        for k in range(self.niter):
            prev = 0
            for i in range(S):
                w = s[i].item()
                if i == S - 1:
                    nxt = 1
                else:
                    nxt = v[i + 1].item()
                wt = self.EMatrix[:, w] + self.TMatrix[prev, :] + self.TMatrix[:, nxt]
                wt = wt[2:] * p
                wt = torch.exp(wt - torch.max(wt))
                v[i] = torch.multinomial(wt, 1) + 2
                if self.mbr and not self.bi:
                    samples[v[i].item()][i] += 1
                elif self.mbr:
                    if k >= 100 and k % 10 == 0:
                        samples[v[i].item()][i] += 1
                prev = v[i].item()
            p += self.step
        if self.mbr:
            return None, None, samples
        prev = 0
        logp = 0.
        for i in range(S):
            w = s[i].item()
            t = v[i].item()
            logp += self.EMatrix[t, w].item() + self.TMatrix[prev, t].item()
            prev = t
        t = v[-1].item()
        logp += self.TMatrix[t, 1].item()
        return v, logp, None
    
    def decode(self, data):
        total_log = 0.
        decoded = []
        for s, _ in zip(data[0], data[1]):
            x, p, _ = self.sampler(s)
            decoded.append(x)
            total_log += p
        total_correct = 0.
        total_len = 0.
        for t, y in zip(data[1], decoded):
            total_correct += t[1:-1].eq(y).sum().item()
            total_len += len(y)
        return total_log, total_correct * 100. / total_len
    
    def posterior_decode(self, data):
        decoded = []
        for s, _ in zip(data[0], data[1]):
            _, _, samples = self.sampler(s)
            decoded.append(samples.argmax(0))
        total_correct = 0.
        total_len = 0.
        for t, y in zip(data[1], decoded):
            total_correct += t[1:-1].eq(y).sum().item()
            total_len += len(y)
        return total_correct * 100. / total_len
    
parser = argparse.ArgumentParser(description='Gibbs Sampler')
parser.add_argument('--par', type=str, default="./",
                    help='location of the parameter matrix')
parser.add_argument('--K', type=int, default=100,
                    help='number of iterations')
parser.add_argument('--beta', type=float, default=1,
                    help='power of conditional probability')
parser.add_argument('--step', type=float, default=0,
                    help='use annealing instead of constant beta')
parser.add_argument('--MBR', action='store_true',
                    help='Minimum Bayes Risk Inference')
parser.add_argument('--burn_in', action='store_true',
                    help='Burn in after 100 steps')
parser.add_argument('--data', type=str, default="./valid.pt",
                    help='location of the data matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
torch.manual_seed(args.seed)

if args.burn_in:
    assert args.MBR and args.K > 100

GSampler = GibbsSampler(args.par, args.K, args.beta, 52, args.step, args.MBR, args.burn_in)
valid = torch.load(args.data)
if args.MBR:
    x = GSampler.posterior_decode(valid)
    print(x)
else:
    x, y = GSampler.decode(valid)
    print(x)
    print(y)