#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 22:29:13 2019

@author: heqingye
"""
from io import open
thefilepath = "./cbt-boundaries.txt"
count = len(open(thefilepath).readlines())
print(count)

from data import Corpus
import numpy as np
corpus = Corpus()
sm, correct = 0, 0
rng = np.random.default_rng()
for line in corpus.label:
    sm += len(line) - 1
    s = np.zeros(len(line) - 1)
    correct += sum(np.equal(line[:-1], s))
    
print((correct, sm))