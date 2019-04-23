#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:22:46 2019

@author: heqingye
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

import data2
import simple_model
import random

from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default="/floyd/input/data/",
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
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

corpus = data2.Corpus(args.data)
ntokens = corpus.dictionary.__len__()

def batchify(data, bsz, pad):
    tensors = []
    msks = []
    prevs = [v[0] for v in data]
    inputs = [v[1][:-1] for v in data]
    targets = [v[1][1:] for v in data]
    seqs = [torch.ones(len(s), dtype=torch.long) for s in inputs]
    nbatch = len(data) // bsz
    for i in range(nbatch):
        y = pad_sequence(prevs[i*bsz : (i+1)*bsz],True,pad)
        x = pad_sequence(inputs[i*bsz : (i+1)*bsz],True,pad)
        t = pad_sequence(targets[i*bsz : (i+1)*bsz],True,pad)
        m = pad_sequence(seqs[i*bsz : (i+1)*bsz],True,pad)
        tensors.append((y.to(device), x.to(device), t.to(device)))
        msks.append(m.to(device))
    return tensors, msks

eval_batch_size = 50
train_data = batchify(corpus.train, args.batch_size, 0)
val_data = batchify(corpus.valid, eval_batch_size, 0)
test_data = batchify(corpus.test, eval_batch_size, 0)

model = simple_model.SimpleRNNModel(ntokens, args.emsize, args.nhid, args.nlayers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

def get_batch(source, i):
    data_source = source[0][i]
    mask_src = source[1][i]
    context = data_source[0]
    data = data_source[1]
    mask = mask_src.float().view(-1)
    target = data_source[2]
    
    return context, data, target, mask

def evaluate(data_source):
    model.eval()
    total_correct = 0.
    total_len = 0.
    with torch.no_grad():
        for i in range(len(data_source[1])):
            hidden = model.init_hidden(eval_batch_size)
            context, data, target, mask = get_batch(data_source, i)
            _, hid = model(context, hidden)
            output, _ = model(data, hid)
            target_flat = target.view(-1)
            label = torch.argmax(output, 1)
            comp = torch.eq(label, target_flat).float()
            total_correct += (comp * mask).sum()
            total_len += mask.sum()
    
    return total_correct * 100. / (total_len)

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    tr = train_data[1]
    lst = list(range(len(tr)))
    random.shuffle(lst)
    for i in lst:
        context, data, target, _ = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = model.init_hidden(args.batch_size)
        optimizer.zero_grad()
        _, hid = model(context, hidden)
        output, _ = model(data, hid)
        loss = criterion(output, target.view(-1))
        loss.backward()
        

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
#        for p in model.parameters():
#            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | val acc {:2.2f}'.format(
                epoch, i, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), evaluate(val_data)))
            model.train()
            total_loss = 0
            start_time = time.time()
            
best_val_acc = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_acc = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:2.2f}s | valid acc {:2.2f} | '.format(epoch, (time.time() - epoch_start_time),
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
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_acc = evaluate(test_data)
print('=' * 89)
print('| End of training | test acc {:2.2f}'.format(
    test_acc))
print('=' * 89)