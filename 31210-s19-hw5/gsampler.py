#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 23:13:33 2019

@author: heqingye
"""
import numpy as np
from collections import Counter
from data import Corpus

class GibbsSampler:
    
    def __init__(self, gamma, s, beta, corpus, tau=1, step=0.01, pchar=0, K=100, seed=20191229, tol=1e-4):
        self.corpus = corpus
        self.niter = K
        self.s = s
        self.dic = Counter()
        self.beta = beta
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.tol = tol
        self.pchar = pchar
        if pchar == 0:
            self.N = len(self.corpus.unigram)
        else:
            self.N = sum(self.corpus.unigram.values())
        self.t = tau
        if tau < 1:
            self.step = step
        else:
            self.step = 0
        
    def G0(self, s):
        if self.pchar == 0:
            return np.exp((len(s) - 1) * np.log(1 - self.beta) + np.log(self.beta) - len(s) * np.log(self.N))
        if self.pchar == 1:
            return np.exp((len(s) - 1) * np.log(1 - self.beta) + np.log(self.beta) + sum(np.log(self.corpus.unigram[x]) for x in s) - len(s) * np.log(self.N))
        if self.pchar == 2:
            return np.exp((len(s) - 1) * np.log(1 - self.beta) + np.log(self.beta) + sum(np.log(self.corpus.bigram[s[i:i+2]]) for i in range(len(s) - 1)) - np.log(self.N) - sum(np.log(self.corpus.unigram[x]) for x in s[1:-1]))
        
    def sampler(self):
        total_len, correct = 0, 0
        bds = []
        segments = 0
        res = []
        for j, sentence in enumerate(self.corpus.data):
            S = len(sentence)
            total_len += S - 1
            s = self.rng.binomial(1, self.gamma, S)
            s[-1] = 1
            segments += sum(s)
            correct += sum(np.equal(self.corpus.label[j][:-1], s[:-1]))
            idx = np.nonzero(s)[0]
            bds.append(idx)
            prev = 0
            for i in idx:
                self.dic[sentence[prev:i+1]] += 1
                prev = i + 1
        print("initial BPA: {}".format(correct / total_len))
        res.append(correct / total_len)
        for k in range(self.niter):
            correct = 0
            for j, sentence in enumerate(self.corpus.data):
                S = len(sentence)
                newb = []
                prev, l = 0, 0
                for i in range(S - 1):
                    if i == bds[j][l]:
                        l += 1
                        full, prv, nxt = sentence[prev:bds[j][l]+1], sentence[prev:i+1], sentence[i+1:bds[j][l]+1]
                        p0 = (self.dic[full] + self.s * self.G0(full)) / (segments - 2 + self.s)
                        p1 = (1 - self.gamma) * (self.dic[prv] - 1 + self.s * self.G0(prv)) / (segments - 2 + self.s)
                        p1 *= (self.dic[nxt] - 1 + int(prv == nxt) + self.s * self.G0(nxt)) / (segments - 1 + self.s)
                        s = self.rng.binomial(1, np.power(p1, self.t) / (np.power(p0, self.t) + np.power(p1, self.t)), 1)
                        if s[0]:
                            prev = i + 1
                            newb.append(i)
                            pred = 1
                        else:
                            segments -= 1
                            self.dic[full] += 1
                            self.dic[prv] -= 1
                            self.dic[nxt] -= 1
                            pred = 0
                    else:
                        full, prv, nxt = sentence[prev:bds[j][l]+1], sentence[prev:i+1], sentence[i+1:bds[j][l]+1]
                        p0 = (self.dic[full] - 1 + self.s * self.G0(full)) / (segments - 1 + self.s)
                        p1 = (1 - self.gamma) * (self.dic[prv] + self.s * self.G0(prv)) / (segments - 1 + self.s)
                        p1 *= (self.dic[nxt] + int(prv == nxt) + self.s * self.G0(nxt)) / (segments + self.s)
                        s = self.rng.binomial(1, np.power(p1, self.t) / (np.power(p0, self.t) + np.power(p1, self.t)), 1)
                        if s[0]:
                            prev = i + 1
                            segments += 1
                            self.dic[full] -= 1
                            self.dic[prv] += 1
                            self.dic[nxt] += 1
                            newb.append(i)
                            pred = 1
                        else:
                            pred = 0
                    if self.dic[full] < 0 or self.dic[prv] < 0 or self.dic[nxt] < 0:
                        print("Warning! Negative Counts!")
                    if pred == self.corpus.label[j][i]:
                        correct += 1
                newb.append(S - 1)
#                print(newb)
                bds[j] = np.array(newb)
            self.t += self.step
            print("round {} BPA: {}".format(k + 1, correct / total_len))
            if abs(correct / total_len - res[-1]) < self.tol:
                return res
            res.append(correct / total_len)
        return res
    
    def get_segments(self, topk):
        print(self.dic.most_common(topk))
        
    def get_sentences(self, topk):
        cnt = 0
        for s, l in zip(self.corpus.data, self.corpus.label):
            if cnt > topk:
                break
            ans = []
            prev = 0
            idx = np.nonzero(l)[0]
            for i in idx:
                ans.append(s[prev:i + 1])
                prev = i + 1
            print(" ".join(ans))
            cnt += 1
            
corpus = Corpus()
GSampler = GibbsSampler(0.5, 1, 0.5, corpus, tau=0.1, pchar=2, K=200)
GSampler.sampler()
GSampler.get_segments(200)
GSampler.get_sentences(20)
