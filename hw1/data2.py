#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:59:30 2019

@author: heqingye
"""

import os
from io import open
import torch


class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                word = line.split()[0]
                if word not in self.word2idx:
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)
    
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary(os.path.join(path, 'bobsue.voc.txt'))
        self.train = self.tokenize(os.path.join(path, 'bobsue.prevsent.train.tsv'))
        self.valid = self.tokenize(os.path.join(path, 'bobsue.prevsent.dev.tsv'))
        self.test = self.tokenize(os.path.join(path, 'bobsue.prevsent.test.tsv'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        sentences = []
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                sents = line.strip().split('\t')
                assert len(sents) == 2
                tensors = []
                for sent in sents:
                    words = sent.split()
                    ids = torch.LongTensor(len(words))
                    token = 0
                    for word in words:
                        pos = self.dictionary.word2idx[word]
                        ids[token] = pos
                        token += 1
                    tensors.append(ids)
                sentences.append(tensors)
        
        return sentences