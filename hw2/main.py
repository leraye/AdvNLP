#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:20:07 2019

@author: heqingye
"""

import argparse
import time
#import statistics as stat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

import data
#import selfattention
import mod1
import random

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default="/floyd/input/data/",
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
ntokens = corpus.dictionary.__len__()

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(source, bsz, pad):
    sents = source[0]
    labels = source[1]
    masks = source[2]
    data = []
    bmsk = []
    N = len(sents)
    nbatch = N // bsz
    for i in range(nbatch + 1):
        x = pad_sequence(sents[i * bsz : min((i + 1) * bsz, N)], True, pad)
        m = pad_sequence(masks[i * bsz : min((i + 1) * bsz, N)], True, 0)
        y = torch.LongTensor(labels[i * bsz : min((i + 1) * bsz, N)])
        data.append((x, y))
        bmsk.append(m)
    return data, bmsk

eval_batch_size = 64
train_data = batchify(corpus.train, args.batch_size, ntokens)
val_data = batchify(corpus.valid, eval_batch_size, ntokens)
test_data = batchify(corpus.test, eval_batch_size, ntokens)

#model = mod1.SimpleClassifier(ntokens, args.emsize, args.nhid, args.nlayers, 2).to(device)
#model = mod1.CosAttenClassifier(ntokens, args.emsize, args.nhid, args.nlayers, 2).to(device)
model = mod1.BiLSTMAttention(ntokens, args.emsize, args.nlayers, 2).to(device)
#model = selfattention.ScaledDPAttention(ntokens, args.emsize, args.nlayers, 2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def get_batch(source, i, dev):
    data = source[0][i][0].to(dev)
    target = source[0][i][1].to(dev)
    masks = source[1][i].float().to(dev)
    return data, target, masks

def top_K_norm(k=15, maxormin=True):
    W = model.encoder.weight.data
    x = torch.norm(W, dim=1)
    idx = torch.topk(x, k, largest=maxormin)[1]
    for i in idx:
        print(corpus.dictionary.find_word(i))
        
def top_K_sim(k=15, maxormin=True):
    W = model.encoder.weight.data
    scores = F.cosine_similarity(model.u.unsqueeze(0).expand_as(W), W, 1)[:-1]
    idx = torch.topk(scores, k, largest=maxormin)[1]
    for i in idx:
        print(corpus.dictionary.find_word(i))
    

def evaluate(data_source):
    model.eval()
    total_correct = 0.
    total = 0.
    with torch.no_grad():
        for i in range(len(data_source[1])):
            data, target, mask = get_batch(data_source, i, device)
            output = model(data, mask)
            total += len(target)
            label = torch.argmax(output, 1)
            total_correct += label.eq(target).sum().item()
    return total_correct * 100. / total

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    tr = train_data[1]
    lst = list(range(len(tr)))
    random.shuffle(lst)
    for i in lst:
        data, target, mask = get_batch(train_data, i, device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        
        optimizer.zero_grad()
        output = model(data, mask)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | valid acc {:2.2f}'.format(
                epoch, i, elapsed * 1000 / args.log_interval, cur_loss, evaluate(val_data)))
            model.train()
            total_loss = 0
            start_time = time.time()

best_val_acc = None

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_acc = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(epoch, (time.time() - epoch_start_time),
                                           val_acc))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_acc or val_acc > best_val_acc:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_acc = val_acc
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f}'.format(
    test_loss))
print('=' * 89)

#top_K_norm(15)
#top_K_sim(15)
#print()
#top_K_norm(15, False)
#top_K_sim(15, False)

#tr = train_data[1]
#attn_dict = {}
#for k in range(len(tr)):
#    data, _, mask = get_batch(train_data, k)
#    emb = model.encoder(data) * mask.unsqueeze(2)
#    cossim = torch.exp(F.cosine_similarity(model.u.unsqueeze(0).unsqueeze(1).expand_as(emb), emb, 2)) * mask # bsz x seq
#    attn = cossim / torch.sum(cossim, 1, True)
#    for i in range(data.size(0)):
#        for j in range(data.size(1)):
#            idx = data[i,j].item()
#            if idx < ntokens and idx not in attn_dict:
#                attn_dict[idx] = [attn[i,j].item()]
#            elif idx in attn_dict:
#                attn_dict[idx].append(attn[i,j].item())
                
                
#cv_dict = {}
#for w, c in corpus.freq.items():
#    if c >= 100:
#        wid = corpus.dictionary.find_id(w)
#        l = attn_dict[wid]
#        cv_dict[w] = stat.stdev(l) / stat.mean(l)
#top30 = sorted(cv_dict.items(), key=lambda kv: kv[1], reverse=True)[:30]
#for w, cv in top30:
#    print(w)