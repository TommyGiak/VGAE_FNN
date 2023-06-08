# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:39:16 2023

@author: Tommaso Giacometti
"""

import numpy as np
import torch
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

data = models.Data_Bio(features, pos_edges_train, neg_edges_train, pos_edges_test, neg_edges_test)
#%%

in_channels = data.x.shape[1]
hid_dim = 100
emb_dim = 50
lr = 1e-2
autoencoder = models.VGAE(in_channels, hid_dim, emb_dim).to(device)


#%%
#Training
print(f'{plots.Bcolors.HEADER}Training of the VGAE{plots.Bcolors.ENDC}')
lossi = autoencoder.train_cycle(data, epochs=2000)


#%%
#Plots

plots.plot_loss(lossi)
plots.plot_train_distribution_VGAE(autoencoder, data.x, data.train_pos)
plots.plot_test_distribution_VGAE(autoencoder, data.x, data.train_pos, data.test_pos, data.test_neg)


#%%
#Data processing for the FFNN
embedding = autoencoder(data.x, data.train_pos)[0].detach() #[0] -> To get only z and not logvar

train_emb_pos = models.get_fnn_input(embedding, data.train_pos)
train_emb_neg = models.get_fnn_input(embedding, data.train_neg)
test_emb_pos = models.get_fnn_input(embedding, data.test_pos)
test_emb_neg = models.get_fnn_input(embedding, data.test_neg)


#%%
#FNN
fnn = models.FNN(emb_dim*2).to(device)
lossi = []
lossi_test = []


#%%
print(f'{plots.Bcolors.HEADER}Training of the FNN{plots.Bcolors.ENDC}')
lossi, lossi_test = fnn.train_cycle_fnn(train_emb_pos, train_emb_neg, test_emb_pos, test_emb_neg)

plots.plot_loss(lossi, 10)


#%%
fnn.eval()
link = data.test_neg
inp = models.get_fnn_input(embedding, link)

h = torch.nn.functional.softmax(fnn(inp), dim = 1)
x = h.detach().cpu().numpy()[:,0]
y = h.detach().cpu().numpy()[:,1]

fig, ax  = plt.subplots()
ax.hist(x, bins=30)
plt.show()










