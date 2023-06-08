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
from plots import Bcolors
import matplotlib.pyplot as plt


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
data = models.Data_Papers(data)


#%%
#Creation of the model
in_channels = data.x.shape[1]
hid_dim = 100
emb_dim = 50
autoencoder = models.VGAE(in_channels, hid_dim, emb_dim).to(device)


#%%
#Training
print(f'{plots.Bcolors.HEADER}Training of the VGAE{plots.Bcolors.ENDC}')
lossi = autoencoder.train_cycle(data)

    
#%%
#Plots

plots.plot_loss(lossi)
plots.plot_train_distribution_VGAE(autoencoder, data.x, data.train_pos)
plots.plot_test_distribution_VGAE(autoencoder, data.x, data.train_pos, data.test_pos, data.test_neg)


#%%
#Data processing for the fnn
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

plots.plot_loss(lossi, 20)


#%%
fnn.eval()
link = data.test_pos
inp = models.get_fnn_input(embedding, link)

h = torch.nn.functional.softmax(fnn(inp), dim = 1)
x = h.detach().cpu().numpy()[:,0]
y = h.detach().cpu().numpy()[:,1]

fig, ax  = plt.subplots()
ax.hist(x, bins=30)


#%%



    
