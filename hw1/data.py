#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:00:51 2019

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
    def __init__(self, path, NS=False):
        self.dictionary = Dictionary(os.path.join(path, 'bobsue.voc.txt'))
        self.NegSamp = NS
        if NS:
            self.unigram = torch.zeros(self.dictionary.__len__())
        self.train = self.tokenize(os.path.join(path, 'bobsue.lm.train.txt'), True)
#        self.train = self.packup(train, 12)
        self.valid = self.tokenize(os.path.join(path, 'bobsue.lm.dev.txt'))
#        self.valid = self.packup(valid, 50)
        self.test = self.tokenize(os.path.join(path, 'bobsue.lm.test.txt'))
#        self.test = self.packup(test, 50)

    def tokenize(self, path, train=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        sentences = []
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.strip().split()
                ids = torch.LongTensor(len(words))
                token = 0
                for word in words:
                    pos = self.dictionary.word2idx[word]
                    ids[token] = pos
                    token += 1
                    if train and self.NegSamp and pos > 0:
                        self.unigram[pos] += 1
                sentences.append(ids)
        
        return sentences