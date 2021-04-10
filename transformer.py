#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 01:53:31 2020

@author: jingtao
"""

from collections import defaultdict
import os
import pickle
import sys

import numpy as np

from rdkit import Chem
import pickle
import sys
import timeit
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import math
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.autograd import Variable
import matplotlib.pyplot as plt

### hyper parameters

ngram=3
DATASET='yeast'

radius='2'
dim=10
d_ff=10 
layer_output=1
heads=2
n_encoder=3
n_decoder=1
lr=5e-3 
lr_decay=0.5
decay_interval=10 
weight_decay=0
iteration=200 
warmup_step=50
dropout=0.1
batch_size = 32


setting = f'ngram={ngram} DATASET={DATASET} radius={radius}\
dim={dim}\
d_ff={d_ff}\
layer_output={layer_output}\
heads={heads}\
n_encoder={n_encoder}\
n_decoder={n_decoder}\
lr={lr}\
lr_decay={lr_decay}\
decay_interval={decay_interval}\
weight_decay={weight_decay}\
iteration={iteration}\
warmup_step={warmup_step}\
dropout={dropout}\
batch_size = {batch_size}'

### PPI model

class PPI_predictor(nn.Module):
    def __init__(self):
        super(PPI_predictor, self).__init__()
        
        
        ### hyperparameters:        
        self.n_encoder=n_encoder
        self.n_decoder=n_decoder
        self.heads=heads
        self.dropout = dropout
        self.dim=dim
        self.dim_gnn=dim
        self.d_ff=d_ff        
        
        
        
        ##max pooling (k most excited neurons)
        self.k=40
        
        ### models

        # cnn:
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*11+1,
                     stride=1, padding=11) for _ in range(3)])
        self.W_attention = nn.Linear(self.dim, self.dim)
        
        
        ## multi channel
        '''
        self.W_cnn = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=2, 
                      kernel_size=2*window+1,
                     stride=1, padding=window),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=2*window+1,
                     stride=1, padding=window),
            nn.Conv2d(in_channels=2, out_channels=1, 
                      kernel_size=2*window+1,
                     stride=1, padding=window),
            ])
        '''
        
        # transformer:
        self.embed_word = nn.Embedding(n_word, self.dim)

        self.encoder=encoder(self.n_encoder, self.dim, self.d_ff, self.dropout, heads=self.heads)
        self.decoder=decoder(self.n_decoder, self.dim, self.d_ff, self.dropout, heads=self.heads)
        
        self.tgt1=tgt_out( self.heads, self.dim, dropout=0)
        
        ## concate & attention
        self.W_ff1 = nn.Linear(self.dim, self.dim)
        self.W_ff2 = nn.Linear(self.dim, self.dim)
        self.linears = clones(nn.Linear(dim, dim), 4)
        # interaction:
        self.W_out = nn.Linear(self.k, 1)
        self.W_interaction = nn.Linear(2*self.dim, 2)
        
    
    def attention_cnn(self,  xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        #h = torch.relu(self.W_attention(x))
        #hs = torch.relu(self.W_attention(xs))
        #weights = torch.tanh(F.linear(h, hs))
        #ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        #return torch.unsqueeze(torch.mean(ys, 0), 0)
        return xs
        
    def transformer(self, protein, n_encoder, n_decoder, heads):
        protein=self.encoder(protein)
        return protein
        
    def forward(self, inputs): ## inputs=[p1,p2,interaction]

        p1, p2 = inputs


        """Processing each protein representations with PROTEIN TRANSFORMER """
        
        words1 = self.embed_word(p1)
        words2 = self.embed_word(p2)
        
        words1=self.attention_cnn(words1,3)
        words2=self.attention_cnn(words2,3)
        
        p1_vector = self.transformer(words1,
                                     self.n_encoder,self.n_decoder,self.heads)
        p2_vector = self.transformer(words2,
                                     self.n_encoder,self.n_decoder,self.heads)
        # attention CNN
        #protein_vector = self.attention_cnn(compound_vector,protein_vector,3)
        
        """Concatenate the above two vectors using attention mechanism and output the interaction."""

        #print(p1_vector.size())
  

        p1=p1_vector.topk(self.k,dim=0).values
        p2=p2_vector.topk(self.k,dim=0).values
        
        p1=self.W_ff1(p1)  # k x dim
        p2=self.W_ff2(p2)  # k x dim
        
        
        nwords = self.k
        d_k=int(self.dim/self.heads)
        query, key, value1, value2 = \
            [l(x).view(nwords, -1, self.heads, d_k).transpose(1, 2)
             for l, x in zip(self.linears, (p1, p2, p1,p2))]
        # qkv.size() = length,heads,1,dk
            
        query=query.squeeze(2).transpose(0,1)               # heads, length, dk
        key=key.squeeze(2).transpose(0,1).transpose(1,2)    # heads, dk, length
        value1=value1.squeeze(2).transpose(0,1)               # heads, length, dk
        value2=value2.squeeze(2).transpose(0,1)  
        scores = torch.matmul(query,key)                    # heads, length, length
        p_attn = F.softmax(scores, dim = 2)                 # heads, length, length


        
        x1=torch.matmul(p_attn, value1)                       # heads, length, dk
        x1=x1.transpose(0,1).contiguous().view([nwords,self.heads * d_k]) 
        x1=self.linears[-1](x1) # k x dim
        x1=x1.sum(dim=0).view([1,self.dim])
        
        
        x2=torch.matmul(p_attn, value2)                       # heads, length, dk
        x2=x2.transpose(0,1).contiguous().view([nwords,self.heads * d_k]) 
        x2=self.linears[-1](x2)
        x2=x2.sum(dim=0).view([1,self.dim])
        
        
        cat_vector = torch.cat((x1, x2), 1)
        
        #print(cat_vector.size())
        
        interaction = self.W_interaction(cat_vector)
        
        return interaction,p1_vector,p2_vector
    
    def attnscore(self,p1): ## probably not useful because the model has changed
        words1 = self.embed_word(p1)
        
        words1=self.attention_cnn(words1,3)
        
        p1_vector = self.transformer(words1,
                                     self.n_encoder,self.n_decoder,self.heads)
        scores = torch.matmul(p1_vector,p1_vector.T)                    # heads, length, length
        p_attn = F.softmax(scores, dim = 1)
        fig, ax = plt.subplots()
        im = ax.imshow(p_attn.detach())
        print(p_attn)
        
        
    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        #print(inputs)
        #print(correct_interaction)
        predicted_interaction,p1,p2 = self.forward(inputs)

        if train:
            #print(predicted_interaction.size())
            #print(correct_interaction.size())
            #print(correct_interaction)
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

# end of the model


### transformer:
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class encoder(nn.Module):
    def __init__(self, n, dim, d_ff, dropout, heads):
        super(encoder, self).__init__()
        self.layers = clones(encoder_layer(dim, heads, 
                             self_attn(heads, dim, dropout).to(device), 
                             PositionwiseFeedForward(dim, d_ff), dropout), n)
        
    def forward(self, x, mask=None): 
        for layer in self.layers:
            x = layer(x, mask)
        return x

class decoder(nn.Module):
    def __init__(self, n, dim, d_ff, dropout, heads):
        super(decoder, self).__init__()
        self.layers = clones(decoder_layer(dim, heads, 
                             tgt_attn(heads, dim,dropout).to(device), 
                             self_attn(heads, dim, dropout).to(device),
                             PositionwiseFeedForward(dim, d_ff), dropout), n)
        self.tgt_out = tgt_out(heads, dim, dropout)
        self.final_norm = LayerNorm(dim)
    def forward(self, x, tgt):
        for layer in self.layers:
            x = layer(x, tgt)
        x=self.tgt_out(tgt,x,x)
        x=self.final_norm(x)
        return x

## attentions:
class self_attn(nn.Module):
    def __init__(self, h, dim, dropout=0):
        super(self_attn, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = dim // h
        self.h = h
        self.linears = clones(nn.Linear(dim, dim), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None): 
        if mask is not None:
            mask = mask.unsqueeze(1)
        nwords = key.size(0) 
      
        query, key, value = \
            [l(x).view(nwords, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # qkv.size() = length,heads,1,dk
            
        query=query.squeeze(2).transpose(0,1)               # heads, length, dk
        key=key.squeeze(2).transpose(0,1).transpose(1,2)    # heads, dk, length
        value=value.squeeze(2).transpose(0,1)               # heads, length, dk
        
        scores = torch.matmul(query,key)                    # heads, length, length
        p_attn = F.softmax(scores, dim = 2)                 # heads, length, length
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        
        x=torch.matmul(p_attn, value)                       # heads, length, dk
        x=x.transpose(0,1).contiguous().view([nwords,self.h * self.d_k]) 
        #x=x.transpose(0,1).view([nwords,self.h * self.d_k]) 
        self.attn=p_attn  
        
        return self.linears[-1](x).unsqueeze(1)  

class tgt_out(nn.Module):    
    def __init__(self, h, dim, dropout=0):
        super(tgt_out, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = dim // h
        self.h = h
        self.tgt_linear = nn.Linear(10,dim)
        self.linears = clones(nn.Linear(dim, dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):   #q=tgt, k=self, v=self 
        nwords = key.size(0) 
        query = self.tgt_linear(query) # from gnn_dim to dim
        query = self.linears[0](query).view(-1,self.h,self.d_k).transpose(0,1)        # heads, 1, dk
        key   = self.linears[1](key).view(nwords,-1,self.h,self.d_k).transpose(1,2)   # length, heads, 1, dk
        value = self.linears[2](value).view(nwords,-1,self.h,self.d_k).transpose(1,2) # length, heads, 1, dk

        key=key.squeeze(2).transpose(0,1).transpose(1,2)    # heads, dk, length
        scores = torch.matmul(query,key) 
        p_attn = F.softmax(scores, dim = 2)     # heads,1,length

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        value=value.squeeze(2).transpose(0,1)   # heads,length,dk 
        
        x=torch.matmul(p_attn, value)         
        x=x.transpose(0,1).contiguous().view([1,self.h * self.d_k]) 
        self.attn=p_attn  
        
        return self.linears[-1](x) 
    
class tgt_attn(nn.Module):    
    def __init__(self, h, dim, dropout=0):
        super(tgt_attn, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = dim // h
        self.h = h
        self.tgt_linear = nn.Linear(10,dim)
        self.linears = clones(nn.Linear(dim, dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):   #q=tgt, k=self, v=self 
        nwords = key.size(0) 
        query = self.tgt_linear(query) # from gnn_dim to dim
        query = self.linears[0](query).view(-1,self.h,self.d_k).transpose(0,1)        # heads, 1, dk
        key   = self.linears[1](key).view(nwords,-1,self.h,self.d_k).transpose(1,2)   # length, heads, 1, dk
        value = self.linears[2](value).view(nwords,-1,self.h,self.d_k).transpose(1,2) # length, heads, 1, dk

        key=key.squeeze(2).transpose(0,1).transpose(1,2)    # heads, dk, length
        scores = torch.matmul(query,key) 
        p_attn = F.softmax(scores, dim = 2).transpose(1,2)     # heads,length,1

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        
        value=value.squeeze(2).transpose(0,1)                  # heads,length,dk
        
        x=p_attn*value       #  heads,length,dk
        x=x.transpose(0,1).contiguous().view([nwords,self.h * self.d_k]) 
        self.attn=p_attn     #  length, dim
        
        return self.linears[-1](x) 
## end of attentions


## encoder & decoder layers
class encoder_layer(nn.Module):
    def __init__(self, dim, heads, self_attn, feedforward, dropout):  
        super(encoder_layer, self).__init__()
        self.res_layer = [residual_layer( dim, dropout,self_attn),
                          residual_layer( dim, dropout,feedforward)]
        self.dim=dim
    def forward(self, x, mask=None):
        x = self.res_layer[0](x,x,x)
        return self.res_layer[1](x)
    
    
class decoder_layer(nn.Module):
    def __init__(self, dim, heads, tgt_attn, self_attn, feedforward, dropout):  
        super(decoder_layer, self).__init__()
        self.res_layer = [residual_layer( dim, dropout,tgt_attn),
                          residual_layer( dim, dropout,self_attn),
                          residual_layer( dim, dropout,feedforward)]

    def forward(self, x,tgt):
        x = self.res_layer[0](x,tgt,x)  # res_layer: v, q, k
        x = self.res_layer[1](x,x,x)        
        return self.res_layer[2](x)
## end of encoder & decoder layers

class residual_layer(nn.Module):
    def __init__(self, size, dropout, sublayer):
        super(residual_layer, self).__init__()
        self.norm = LayerNorm(size).to(device)
        self.dropout = nn.Dropout(dropout)
        self.sublayer=sublayer
        
    def forward(self,x, q=None, k=None ):  # q and k are None if sublayer is ff, x is v
        if (q!=None and k!=None):
            return self.norm(x+self.dropout(self.sublayer(q,k,x).squeeze(1)))
        else:
            return self.norm(x+self.dropout(self.sublayer(x).squeeze(1)))
    
    
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        #return norm+bias
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2   
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff).to(device)
        self.w_2 = nn.Linear(d_ff, d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
### end of transformer


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=1e-7, weight_decay=weight_decay) ##start with warm up rate
        self.batch_size=batch_size
        
    def train(self, dataset):
        
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        loss_batch = 0 
        cnt = 0
        for data in dataset:
            loss = self.model(data)
            loss_batch += loss
            cnt += 1
            
            if (cnt == batch_size):
                cnt = 0
                self.optimizer.zero_grad()
                
                
                (loss_batch/self.batch_size).backward()
                self.optimizer.step()
            
                loss_total += loss_batch.to('cpu').data.numpy()
                loss_batch = 0
            
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        
        tp=np.sum(T)
        total=len(T)
        tn=total-tp
        fp=0
        for i in range(len(T)):
            if Y[i]==[1] and T[i]!=[1]:
                fp+=1
        specificity=tn/(tn+fp)
        
        f1=2*(precision*recall)/(precision+recall)
        
        return AUC, precision, recall, specificity, f1

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def rm_long(dataset,length):
    d=[]
    for i in dataset:
        if (i[2].size()[0]<length):
            d.append(i)
    return d

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2












### data preprocessing


"""CPU or GPU."""
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

## reading dataset
# protein dictionary
dic_file=open('data/yeast_dic.tsv','r')
dic_lines=dic_file.readlines()
dic={}
for i in dic_lines:
    item=i.strip().split()
    dic[item[0]]=item[1]
dic_file.close()

#PPIs
ppi_file=open('data/yeast_ppi.tsv','r')
ppi_lines=ppi_file.readlines()

## constructing dataset
word_dict = defaultdict(lambda: len(word_dict))
def split_sequence(sequence, ngram): ## turning sequence into words
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)

dataset=[]
for i in ppi_lines:
    ppi=i.strip().split()
    p1=dic[ppi[0]]
    p2=dic[ppi[1]]
   
    w1 = split_sequence(p1, ngram)
    #print(w1)
    w2 = split_sequence(p2, ngram)
    #print(w2)
    interaction=[int(ppi[2])]
    #if (interaction!=0 and interaction!=1):
    #    print(interaction)
    #    print('error')
    w1=torch.LongTensor(w1).to(device)
    w2=torch.LongTensor(w2).to(device)
    interaction=torch.LongTensor(interaction).to(device)
    
    #if (interaction!=0 and interaction!=1):
    #    print(ppi[2])
    dataset.append([w1,w2,interaction])  ## int array, int array, int




### datasets:
    
#dataset = rm_long(dataset,6000)
dataset = shuffle_dataset(dataset, 1234)
#dataset=dataset[:50]
dataset_train, dataset_ = split_dataset(dataset, 0.8)
dataset_dev, dataset_test = split_dataset(dataset_, 0.5)


print('Preprocessing finished. Ready for training.')


















### training


n_word = len(word_dict)

"""Set a model."""
torch.manual_seed(1234)
model = PPI_predictor().to(device)
trainer = Trainer(model)
tester = Tester(model)

"""Output files."""
file_AUCs = 'output/result/'+setting+'.txt'
file_model = 'output/model/model'
AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'AUC_test\tPrecision_test\tRecall_test\tspecificity_test\tf1_test')
with open(file_AUCs, 'w') as f:
    f.write(AUCs + '\n')

"""Start training."""
print('Training...')
print(AUCs)
start = timeit.default_timer()


 
for epoch in range(1, warmup_step):
        
    trainer.optimizer.param_groups[0]['lr'] += (lr-1e-7)/warmup_step
    loss_train = trainer.train(dataset_train)
    AUC_dev = tester.test(dataset_dev)[0]
    AUC_test, precision_test, recall_test,specificity_test,f1_test = tester.test(dataset_test)
    end = timeit.default_timer()
    time = end - start
    AUCs = [epoch, time, loss_train, AUC_dev,
             AUC_test, precision_test, recall_test,specificity_test,f1_test]
    tester.save_AUCs(AUCs, file_AUCs)
    tester.save_model(model, file_model)
    print('\t'.join(map(str, AUCs)))
        
for epoch in range(1, iteration):

    if epoch % decay_interval == 0:
        trainer.optimizer.param_groups[0]['lr'] *= lr_decay

    loss_train = trainer.train(dataset_train)
    AUC_dev = tester.test(dataset_dev)[0]
    AUC_test, precision_test, recall_test,specificity_test,f1_test = tester.test(dataset_test)

    end = timeit.default_timer()
    time = end - start

    AUCs = [epoch, time, loss_train, AUC_dev,
            AUC_test, precision_test, recall_test,specificity_test,f1_test]
    tester.save_AUCs(AUCs, file_AUCs)
    tester.save_model(model, file_model)

    print('\t'.join(map(str, AUCs)))
























