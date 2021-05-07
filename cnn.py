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

dim=10
d_ff=10 
layer_output=1
heads=2

lr=1e-3 
lr_decay=0.5
decay_interval=10 
weight_decay=0
iteration=200 
warmup_step=50
dropout=0.1
batch_size = 16
k=55

setting = f'best22CNNngram={ngram} DATASET={DATASET}\
dim={dim}\
d_ff={d_ff}\
layer_output={layer_output}\
heads={heads}\
lr={lr}\
lr_decay={lr_decay}\
decay_interval={decay_interval}\
weight_decay={weight_decay}\
iteration={iteration}\
warmup_step={warmup_step}\
dropout={dropout}\
batch_size = {batch_size}\
k={k}'

### PPI model

class PPI_predictor(nn.Module):
    def __init__(self):
        super(PPI_predictor, self).__init__()
        
        
        ### hyperparameters:        
        self.heads=heads
        self.dropout = dropout
        self.dim=dim
        self.dim_gnn=dim
        self.d_ff=d_ff        
        
        
        
        ##max pooling (k most excited neurons)
        self.k=k
        
        ### models

        # cnn:
        
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*11+1,
                     stride=1, padding=11) for _ in range(3)])
        
        self.W_attention = nn.Linear(self.dim, self.dim)
        
        '''
        ## multi channel
        window=11
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
            ])'''
        
        
        # transformer:
        self.embed_word = nn.Embedding(n_word, self.dim)

        
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
        

        
    def forward(self, inputs): ## inputs=[p1,p2,interaction]

        p1, p2 = inputs


        """Processing each protein representations with PROTEIN TRANSFORMER """
        
        p1_vector = self.embed_word(p1)
        p2_vector = self.embed_word(p2)
        
        p1_vector=self.attention_cnn(p1_vector,3)
        p2_vector=self.attention_cnn(p2_vector,3)
        

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

    #def save_model(self, model, filename):
    #    torch.save(model.state_dict(), filename)

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
dic_file=open('data/'+DATASET+'_dic.tsv','r')
dic_lines=dic_file.readlines()
dic={}
for i in dic_lines:
    item=i.strip().split()
    dic[item[0]]=item[1]
dic_file.close()

#PPIs
ppi_file=open('data/'+DATASET+'_ppi.tsv','r')
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
dataset_train, dataset_test = split_dataset(dataset, 0.8)


aug=[]
for data in dataset_train:
    p1 = data[0]
    p2 = data[1]
    i = data[2]
    aug.append([p2,p1,i])

dataset_train = dataset_train + aug
dataset_train = shuffle_dataset(dataset_train, 1234)


def check(dataset1,dataset2):
    for i in range(len(dataset1)):
        for j in range(len(dataset2)):
            d1=dataset1[i]
            d2=dataset2[j]
            
            p1=d1[0]
            p2=d1[1]
            
            pa=d2[0]
            pb=d2[1]
            
            if (len(p1)==len(pa) and len(p2)==len(pb)):
                if (str(p1)==str(pa) and str(p2)==str(pb)):
                    print(i,j)
            if (len(p1)==len(pb) and len(p2)==len(pa)):
                if (str(p1)==str(pb) and str(p2)==str(pa)):
                    print(i,j,"qwe")
    

'''
for i in range(len(dataset_test)):
    dataset_test[i][2] = torch.tensor([1]).to(device)-dataset_test[i][2]
'''
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
file_model = 'output/model/cnn_model'+setting
AUCs = ('Epoch\tTime(sec)\tLoss_train\t'
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
    AUC_test, precision_test, recall_test,specificity_test,f1_test = tester.test(dataset_test)
    end = timeit.default_timer()
    time = end - start
    AUCs = [epoch, time, loss_train, 
             AUC_test, precision_test, recall_test,specificity_test,f1_test]
    tester.save_AUCs(AUCs, file_AUCs)
    #tester.save_model(model, file_model)
    print('\t'.join(map(str, AUCs)))
        

best_auc=0
best_itr=0
for epoch in range(1, iteration):

    if epoch % decay_interval == 0:
        trainer.optimizer.param_groups[0]['lr'] *= lr_decay

    loss_train = trainer.train(dataset_train)
    AUC_test, precision_test, recall_test,specificity_test,f1_test = tester.test(dataset_test)
    '''
    if (AUC_test > best_auc):
        best_auc=AUC_test
        torch.save(best_model.state_dict(), file_model)
        best_itr=epoch'''
    
    end = timeit.default_timer()
    time = end - start

    AUCs = [epoch, time, loss_train, 
            AUC_test, precision_test, recall_test,specificity_test,f1_test]
    tester.save_AUCs(AUCs, file_AUCs)
    #tester.save_model(model, file_model)

    print('\t'.join(map(str, AUCs)))
'''
torch.save(best_model.state_dict(), file_model)
print('The best epoch is: '+str(best_itr))

'''



















