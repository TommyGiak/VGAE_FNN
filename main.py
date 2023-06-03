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



#Define device, if gpu is available il will be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Get directory path
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = current_dir + '/data'

#Download data (if not already present in the specific folder)
dataset = Planetoid(root=data_dir, name='Cora')
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
hid_dim = 32
emb_dim = 16
lr = 1e-2
model = models.VGAE(in_channels, hid_dim, emb_dim)


#%%
#Training
epochs = 500
norm = 1/data.x.shape[0]

lossi = model.train_cycle(data.x, train_pos, test_pos, test_neg)

    
#%%
#Plots

plots.plot_loss(lossi)
plots.plot_train_distribution(model, data.x, train_pos)
plots.plot_test_distribution(model, data.x, train_pos, test_pos, test_neg)









