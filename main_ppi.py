#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:39:16 2023

@author: tommygiak
"""

import numpy as np
import torch
import torch_geometric as pyg
import models
import plots
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(5)

#Data processing
features = np.loadtxt('data/features.txt', dtype=np.float32)
pos_edges = np.loadtxt('data/PositiveEdges.txt', dtype=np.int64)
neg_edges = np.loadtxt('data/NegativeEdges.txt', dtype=np.int64)

features = torch.from_numpy(features)
soft = torch.nn.Softmax(dim = 1)
features = soft(features)
pos_edges = torch.from_numpy(pos_edges)
neg_edges = torch.from_numpy(neg_edges)

idx_pos = torch.randperm(pos_edges.shape[0])
idx_neg = torch.randperm(neg_edges.shape[0])
pos_edges = pos_edges[idx_pos].t()
neg_edges = neg_edges[idx_neg].t()

ind_pos = int(pos_edges.shape[1]*0.85)
ind_neg = int(neg_edges.shape[1]*0.85)

pos_edges_train = pos_edges[:,:ind_pos]
neg_edges_train = neg_edges[:,:ind_neg]
pos_edges_test = pos_edges[:,ind_pos:]
neg_edges_test = neg_edges[:,ind_neg:]


#%%

class DataStructure():
    def __init__(self, features, pos_edges_train, neg_edges_train, pos_edges_test, neg_edges_test):
        self.x = features
        self.train_pos = pos_edges_train
        self.train_neg = neg_edges_train
        self.test_pos = pos_edges_test
        self.test_neg = neg_edges_test


data = DataStructure(features, pos_edges_train, neg_edges_train, pos_edges_test, neg_edges_test)
#%%

in_channels = data.x.shape[1]
hid_dim = 100
emb_dim = 50
lr = 1e-2
autoencoder = models.VGAE(in_channels, hid_dim, emb_dim).to(device)


#%%
#Training
lossi = autoencoder.train_cycle(data, epochs=2000)


#%%
#Plots

plots.plot_loss(lossi)
plots.plot_train_distribution(autoencoder, data.x, data.train_pos)
plots.plot_test_distribution(autoencoder, data.x, data.train_pos, data.test_pos, data.test_neg)


#%%
#Data processing for the FFNN
embedding = autoencoder(data.x, data.train_pos)[0].detach() #[0] -> To get only z and not logvar

train_emb = models.get_ffnn_input(embedding, data.train_pos)
neg = models.get_ffnn_input(embedding, data.train_neg)
test_emb_pos = models.get_ffnn_input(embedding, data.test_pos)
test_emb_neg = models.get_ffnn_input(embedding, data.test_neg)


#%%
#FFNN
ffnn = models.FFNN(emb_dim*2).to(device)


#%%
#FFNN training
epochs = 1000
batch_size = 128
optim = torch.optim.Adam(ffnn.parameters(),lr = 1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
index_pos = np.random.randint(0, train_emb.shape[0]-batch_size, epochs)
index_neg = np.random.randint(0, neg.shape[0]-batch_size, epochs)
lossi = []
lossi_test = []

#%%
#Functions for test and train of FFNN
def train(train_emb, neg, i, index_pos, index_neg, loss_fn):
    ffnn.train()
    one = torch.ones(batch_size, dtype=torch.long).to(device)
    zero = torch.zeros(batch_size, dtype=torch.long).to(device)
    optim.zero_grad()
    out_pos = ffnn(train_emb[index_pos[i]:index_pos[i]+batch_size])
    out_neg = ffnn(neg[index_neg[i]:index_neg[i]+batch_size])
    loss = loss_fn(out_pos, one) + loss_fn(out_neg, zero)
    loss.backward()
    optim.step()
    return loss.item()

def test(test_emb_pos, test_emb_neg, loss_fn):
    lossi_test 
    ffnn.eval()
    with torch.no_grad():
        one = torch.ones(test_emb_pos.shape[0], dtype=torch.long).to(device)
        zero = torch.zeros(test_emb_neg.shape[0], dtype=torch.long).to(device)
        out_pos = ffnn(test_emb_pos)
        out_neg = ffnn(test_emb_neg)
        loss = loss_fn(out_pos, one) + loss_fn(out_neg, zero)
        return loss.item()


#%%
for i, ind in enumerate(index_pos):
    lossi.append(train(train_emb, neg, i, index_pos, index_neg, loss_fn))
    lossi_test.append(test(test_emb_pos, test_emb_neg, loss_fn))

print(lossi[-1], lossi_test[-1])

plots.plot_loss(lossi, 10)
plots.plot_loss(lossi_test, 10)


#%%
ffnn.eval()
link = data.test_neg
inp = models.get_ffnn_input(embedding, link)

h = torch.nn.functional.softmax(ffnn(inp), dim = 1)
x = h.detach().cpu().numpy()[:,0]
y = h.detach().cpu().numpy()[:,1]

fig, ax  = plt.subplots()
ax.hist(x, bins=30)
plt.show()










