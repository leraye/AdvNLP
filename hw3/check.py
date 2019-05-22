#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:40:42 2019

@author: heqingye
"""
import data
from hmmpar import SmoothedMLE

path = "../31210-s19-hw3/"
corpus = data.Corpus(path)

MLE = SmoothedMLE(corpus)

MLE.check()

logprob = 0.
sentences, tags = corpus.valid
for s, t in zip(sentences, tags):
    N = len(t)
    for i in range(1, N):
        prev = t[i - 1].item()
        tg = t[i].item()
        logprob += MLE.transition_prob(prev, tg)
        if i < N - 1:
            w = s[i - 1].item()
            logprob += MLE.emission_prob(tg, w)
    
print(logprob)