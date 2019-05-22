#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:20:21 2019

@author: heqingye
"""

import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.tag2idx = {"<s>":0,"</s>":1}
        self.idx2tag = ["<s>","</s>"]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
    
    def add_tag(self, tag):
        if tag not in self.tag2idx:
            self.idx2tag.append(tag)
            self.tag2idx[tag] = len(self.idx2tag) - 1
    
    def find_word(self, idx):
        return self.idx2word[idx]
    
    def find_tidx(self, tag):
        return self.tag2idx[tag]
    
    def find_tag(self, idx):
        return self.idx2tag[idx]

    def __len__(self):
        return len(self.idx2word)
    
    def ntags(self):
        return len(self.idx2tag)
    
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'en_ewt.train'), './train.pt')
        self.valid = self.tokenize(os.path.join(path, 'en_ewt.dev'), './valid.pt')

    def tokenize(self, path, save):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                if len(words) > 0:
                    self.dictionary.add_word(words[0])
                    self.dictionary.add_tag(words[1])

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            sentences = []
            lables = []
            sentence = []
            tags = [0]
            for line in f:
                words = line.split()
                if len(words) > 0:
                    sentence.append(self.dictionary.word2idx[words[0]])
                    tags.append(self.dictionary.tag2idx[words[1]])
                else:
                    tags.append(1)
                    sentences.append(torch.LongTensor(sentence))
                    lables.append(torch.LongTensor(tags))
                    sentence = []
                    tags = [0]
        torch.save((sentences, lables), save)

        return sentences, lables