# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:06:36 2023

@author: Tommaso Giacometti
"""
import os
import torch_geometric as pyg
from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.utils import train_test_split_edges
import models
import plots
import numpy as np



#Define device, if gpu is available il will be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Get directory path
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = current_dir + '/data'

#Download data (if not already present in the specific folder)
norm = pyg.transforms.NormalizeFeatures()
dataset = Planetoid(root=data_dir, name='Cora', transform=norm)
data = dataset[0].to(device)

torch.manual_seed(0)

#Data preprocessing
data = train_test_split_edges(data, 0.05, 0.1)
train_pos = data.train_pos_edge_index
val_pos = data.val_pos_edge_index
val_neg = data.val_neg_edge_index
test_pos = data.test_pos_edge_index
test_neg = data.test_neg_edge_index


#%%
#Creation of the model
in_channels = data.x.shape[1]
hid_dim = 100
emb_dim = 50
lr = 1e-2
autoencoder = models.VGAE(in_channels, hid_dim, emb_dim)


#%%
#Training
lossi = autoencoder.train_cycle(data.x, train_pos, test_pos, test_neg)

    
#%%
#Plots

plots.plot_loss(lossi)
plots.plot_train_distribution(autoencoder, data.x, train_pos)
plots.plot_test_distribution(autoencoder, data.x, train_pos, test_pos, test_neg)


#%%
#Data processing for the FFNN
embedding = autoencoder(data.x, train_pos)[0].detach() #[0] -> To get only z and not logvar

train_emb = models.get_ffnn_input(embedding, train_pos)
test_emb_pos = models.get_ffnn_input(embedding, test_pos)
test_emb_neg = models.get_ffnn_input(embedding, test_neg)

train_emb_neg = test_emb_neg



#%%
#FFNN

ffnn = models.FFNN(emb_dim*2).to(device)

epochs = 200
batch_size = 128
optim = torch.optim.Adam(ffnn.parameters(),lr = 1e-3)
ffnn.train()
loss_fn = torch.nn.CrossEntropyLoss()
index = np.random.randint(0, train_emb.shape[0]-batch_size, epochs)

one = torch.ones(batch_size, dtype=torch.long).to(device)
zero = torch.zeros(batch_size, dtype=torch.long).to(device)

lossi = []

for i, ind in enumerate(index):
    optim.zero_grad()

    # neg = pyg.utils.negative_sampling(train_pos, data.x.shape[0], batch_size)
    # neg = models.get_ffnn_input(embedding, neg)

    out_pos = ffnn(train_emb[ind:ind+batch_size])
    #out_neg = ffnn(neg)
    loss = loss_fn(out_pos, one) #+ loss_fn(out_neg, zero)
    loss.backward()
    optim.step()
    lossi.append(loss.item())

print(loss.item())
plots.plot_loss(lossi, 100)

with torch.no_grad():
    one = torch.ones(test_emb_pos.shape[0], dtype=torch.long).to(device)
    zero = torch.zeros(test_emb_neg.shape[0], dtype=torch.long).to(device)
    out_pos = ffnn(test_emb_pos)
    out_neg = ffnn(test_emb_neg)
    loss = loss_fn(out_pos, one) + loss_fn(out_neg, zero)
    print(loss.item())


#%%
link = test_pos.T[0:2]
inp = models.get_ffnn_input(embedding, link)

torch.nn.functional.softmax(ffnn(inp[1]), dim = 0)


