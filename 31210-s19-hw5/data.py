#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:08:17 2019

@author: heqingye
"""

import os
from io import open
import numpy as np
from collections import Counter

class Corpus(object):
    def __init__(self, path="./"):
        self.unigram = Counter()
        self.bigram = Counter()
        self.data = self.vectorize(os.path.join(path, "cbt-characters.txt"), False)
        self.label = self.vectorize(os.path.join(path, "cbt-boundaries.txt"))

    def vectorize(self, path, label=True):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        ret = []
        if label:
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    v = [int(x) for x in line.strip()]
                    ret.append(np.array(v))
        else:
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    l = line.strip()
                    for i in range(len(l)):
                        self.unigram[l[i]] += 1
                        if i < len(l) - 1:
                            self.bigram[l[i:i+2]] += 1
                    ret.append(l)
        return ret