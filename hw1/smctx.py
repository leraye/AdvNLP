#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:40:32 2019

@author: heqingye
"""

import argparse
import time
import torch

import data
import model
import random
import torch.optim as optim
import torch.nn.functional as F

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
parser.add_argument('--neg', type=int, default=20, metavar='N',
                    help='number of negative samples')
parser.add_argument('--f', type=float, default=0,
                    help='power')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--negsample', action='store_true',
                    help='use binary log loss with negative sampling')
parser.add_argument('--hinge', action='store_true',
                    help='use hinge loss with negative sampling')
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
assert (not args.negsample) or (not args.hinge)

###############################################################################
# Load data
###############################################################################
if args.negsample or args.hinge:
    corpus = data.Corpus(args.data, True)
    probs = corpus.unigram / corpus.unigram.sum()
else:
    corpus = data.Corpus(args.data)
    probs = None

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

def batchify(data, bsz, pad):
    msks = []
    tensors = []
#    data.sort(key=lambda v : len(v), reverse=True)
    inputs = [v[:-1] for v in data]
    targets = [v[1:] for v in data]
    masks = [torch.ones(len(v) - 1, dtype=torch.long) for v in data]
    nbatch = len(data) // bsz
    for i in range(nbatch):
        x = pad_sequence(inputs[i*bsz : (i+1)*bsz],True,pad)
        t = pad_sequence(targets[i*bsz : (i+1)*bsz],True,pad)
        q = pad_sequence(masks[i*bsz : (i+1)*bsz],True,pad)
        tensors.append((x.to(device), t.to(device)))
        msks.append(q.to(device))
    return tensors, msks

eval_batch_size = 50
train_data = batchify(corpus.train, args.batch_size, 0)
val_data = batchify(corpus.valid, eval_batch_size, 0)
test_data = batchify(corpus.test, eval_batch_size, 0)

###############################################################################
# Build the model
###############################################################################

model = model.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, device, args.neg, args.f, probs).to(device)
optimizer = optim.Adam(model.parameters())

def log_loss(score, target, mask):
    normalized_score = F.log_softmax(score, 2).neg()
    set_target = torch.zeros_like(normalized_score) # bsz x seq_len x ntokens
    t = target.unsqueeze(2).expand_as(normalized_score)
    los = torch.sum(normalized_score * set_target.scatter_(2, t, 1), 2) * mask
    return torch.sum(los)

def binary_log_loss(psc, nsc, mask):
    scr = F.logsigmoid(psc.squeeze(1)).neg() - F.logsigmoid(nsc.neg()).mean(1)
    return torch.sum(scr * mask.view(-1))

def hinge_loss(psc, nsc, mask):
    scr = F.relu(1 + psc.neg().expand_as(nsc) + nsc).sum(1)
    return torch.sum(scr * mask.view(-1))

###############################################################################
# Training code
###############################################################################


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    data_source = source[0][i]
    mask_src = source[1][i]
    data = data_source[0]
    mask = mask_src.float()
    target = data_source[1]
    
    return data, target, mask
    
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_correct = 0.
    total_len = 0.
    with torch.no_grad():
        for i in range(len(data_source[0])):
            hidden = model.init_hidden(eval_batch_size)
            data, targets, mask = get_batch(data_source, i)
            output = model.evaluate(data, hidden)
            target_flat = targets.view(-1)
            label = torch.argmax(output, 2).view(-1)
            comp = torch.eq(label, target_flat).float()
            total_correct += (comp * mask.view(-1)).sum().item()
            total_len += mask.sum().item()
    
    return total_correct * 100. / (total_len)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    tr = train_data[0]
    lst = list(range(len(tr)))
    random.shuffle(lst)
    for i in lst:
        data, targets, mask = get_batch(train_data, i)
        hidden = model.init_hidden(args.batch_size)
        optimizer.zero_grad()
        out, sp, sn = model(data, targets, hidden)
        if args.negsample:
            loss = binary_log_loss(sp, sn, mask)
        elif args.hinge:
            loss = hinge_loss(sp, sn, mask)
        else:
            loss = log_loss(out, targets, mask)
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | val acc {:2.2f}'.format(
                epoch, i, elapsed * 1000 / args.log_interval, cur_loss, evaluate(val_data)))
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