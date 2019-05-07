#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:59:41 2019

@author: heqingye
"""

import os
from io import open
import torch
from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.word2idx = {"<unk>":0}
        self.idx2word = ["<unk>"]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    
    def find_id(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return 0
        
    def find_word(self, idx):
        return self.idx2word[idx]


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.freq = Counter()
        self.train = self.tokenize(os.path.join(path, 'senti.train.tsv'), True)
        self.valid = self.tokenize(os.path.join(path, 'senti.dev.tsv'))
        self.test = self.tokenize(os.path.join(path, 'senti.test.tsv'))
        

    def tokenize(self, path, train=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        if train:
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    splitted_line = line.strip().split('\t')
                    assert len(splitted_line) == 2
                    words = splitted_line[0].split()
                    for word in words:
                        self.dictionary.add_word(word)
                        self.freq[word] += 1
                
        # Tokenize file content
        sentences = []
        labels = []
        masks = []
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                splitted_line = line.strip().split('\t')
                assert len(splitted_line) == 2
                words = splitted_line[0].split()
                n = len(words)
                ids = torch.LongTensor(n)
                labels.append(int(splitted_line[1]))
                masks.append(torch.ones(n, dtype=torch.long))
                token = 0
                for word in words:
                    ids[token] = self.dictionary.find_id(word)
                    token += 1
                sentences.append(ids)

        return sentences, labels, masks