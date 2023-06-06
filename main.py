# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:06:36 2023

@author: Tommaso Giacometti
"""
import os
import torch_geometric as pyg
from torch_geometric.datasets import Planetoid
import torch
import models
import plots
import matplotlib.pyplot as plt
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
data = models.DataProcessing(data)


#%%
#Creation of the model
in_channels = data.x.shape[1]
hid_dim = 100
emb_dim = 50
autoencoder = models.VGAE(in_channels, hid_dim, emb_dim).to(device)


#%%
#Training
lossi = autoencoder.train_cycle(data)

    
#%%
#Plots

plots.plot_loss(lossi)
plots.plot_train_distribution_VGAE(autoencoder, data.x, data.train_pos)
plots.plot_test_distribution_VGAE(autoencoder, data.x, data.train_pos, data.test_pos, data.test_neg)


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
epochs = 10000
batch_size = 128
optim = torch.optim.Adam(ffnn.parameters(),lr = 1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
index = np.random.randint(0, train_emb.shape[0]-batch_size, epochs)
lossi = []
lossi_test = []

#%%
#Functions for test and train of FFNN
def train(train_emb, neg, ind, loss_fn):
    ffnn.train()
    one = torch.ones(batch_size, dtype=torch.long).to(device)
    zero = torch.zeros(batch_size, dtype=torch.long).to(device)
    optim.zero_grad()
    out_pos = ffnn(train_emb[ind:ind+batch_size])
    out_neg = ffnn(neg[ind:ind+batch_size])
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
for i, ind in enumerate(index):
    lossi.append(train(train_emb, neg, ind, loss_fn))
    lossi_test.append(test(test_emb_pos, test_emb_neg, loss_fn))

print(lossi[-1], lossi_test[-1])

plots.plot_loss(lossi, 10)
plots.plot_loss(lossi_test, 10)


#%%
ffnn.eval()
link = data.val_pos
inp = models.get_ffnn_input(embedding, link)

h = torch.nn.functional.softmax(ffnn(inp), dim = 1)
x = h.detach().cpu().numpy()[:,0]
y = h.detach().cpu().numpy()[:,1]

fig, ax  = plt.subplots()
ax.hist(x, bins=30)


#%%




