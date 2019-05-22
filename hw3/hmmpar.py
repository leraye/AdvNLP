#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:32:34 2019

@author: heqingye
"""
import torch

# MLE with smoothing
class SmoothedMLE(object):
    def __init__(self, corpus, transit = 0.1, emission = 0.001):
        self.total_tags = corpus.dictionary.ntags()
        self.total_words = corpus.dictionary.__len__()
        self.transition_matrix = torch.zeros(self.total_tags, self.total_tags)
        self.emission_matrix = torch.zeros(self.total_tags, self.total_words)
        self.dict = corpus.dictionary
        self.pt = transit
        self.pe = emission
        self.build_table(corpus.train)
        
    def build_table(self, data):
        for s, t in zip(data[0], data[1]):
            N = len(t)
            for i in range(1, N):
                prev = t[i - 1].item()
                curr = t[i].item()
                self.transition_matrix[prev][curr] += 1
                if i < N - 1:
                    w = s[i - 1].item()
                    self.emission_matrix[curr][w] += 1
        self.transition_matrix += self.pt
        self.emission_matrix += self.pe
        y = torch.sum(self.transition_matrix, 1, True) - self.pt
        y[0, 0] -= self.pt
        y[1, 0] -= self.pt
        z = torch.sum(self.emission_matrix, 1, True)
        self.transition_matrix = torch.log(self.transition_matrix) - torch.log(y)
        self.emission_matrix = torch.log(self.emission_matrix) - torch.log(z)
        torch.save(self.transition_matrix, 'tmatrix.pt')
        torch.save(self.emission_matrix, 'ematrix.pt')
    
    def check(self):
        v, idx = torch.topk(self.transition_matrix[0,], 5)
        for i in range(5):
            print(self.dict.find_tag(idx[i].item()))
            print(torch.exp(v[i]).item())
            
        tidx = self.dict.find_tidx("JJ")
        v1, idx1 = torch.topk(self.emission_matrix[tidx,], 10)
        for i in range(10):
            print(self.dict.find_word(idx1[i].item()))
            print(torch.exp(v1[i]).item())
            
    def transition_prob(self, t1, t2):
        return self.transition_matrix[t1, t2].item()
            
    def emission_prob(self, t, w):
        return self.emission_matrix[t, w].item()