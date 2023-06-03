#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:21:48 2023

@author: Tommaso Giacometti
"""
import torch
import torch_geometric as pyg
from torch import nn


class GCNEncoder(nn.Module):
    
    def __init__(self, in_channels : int, hid_dim : int, emb_dim : int):
        super().__init__()
        self.conv = pyg.nn.GCNConv(in_channels, hid_dim)
        self.mu = pyg.nn.GCNConv(hid_dim, emb_dim)
        self.logvar = pyg.nn.GCNConv(hid_dim, emb_dim)
        
        
    def forward(self, x, edges):
        x = self.conv(x,edges).relu()
        return self.mu(x,edges), self.logvar(x,edges)
    
    
    
    
class VGAE(pyg.nn.VGAE):
    
    def __init__(self, in_channels, hid_dim, emb_dim):
        super().__init__(encoder=GCNEncoder(in_channels, hid_dim, emb_dim))
    
        
    def train_step(self, x, edges, optimizer = None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(),lr=1e-2)
        norm = 1/x.shape[0]
        self.train()
        optimizer.zero_grad()
        z = self.encode(x,edges)
        loss = self.recon_loss(z,edges) + self.kl_loss()*norm
        loss.backward()
        optimizer.step()
        return float(loss)
    
    
    def test_step(self, x, train_pos, test_pos, test_neg):
        self.eval()
        with torch.no_grad():
            z = self.encode(x, train_pos)
            auc, ap = self.test(z, test_pos, test_neg)
            print(f'AUC: {auc:.4f} | AP: {ap:.4f}')
            
    
    def train_cycle(self, x, train_pos, test_pos, test_neg, epochs = 1000, optimizer = None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(),lr=1e-2)
        lossi = []
        for i in range(epochs):
            lossi.append(self.train_step(x, train_pos, optimizer))
            if i%(epochs/20) == 0:
                print(f'{i/epochs*100:.2f}% | loss = {lossi[i]:.4f} -> ', end = '')
                self.test_step(x, train_pos, test_pos, test_neg)

        print(f'100.00% | loss = {lossi[i]:.4f} -> ', end = '')
        self.test_step(x, train_pos, test_pos, test_neg)
        return lossi
        
        

class  FFNN(nn.Module):
    
    def __init__(self, inputs, out = 2):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(inputs, 20), nn.ReLU(),
                                 nn.Linear(20, 10), nn.ReLU(),
                                 nn.Linear(10, out))
        
        
    def forward(self,x):
        logits = self.seq(x)
        return logits
        
    
def get_ffnn_input(embedding, links):
    x = torch.hstack((embedding[0,:],embedding[1,:]))
    for i in links.T:
        row = torch.hstack((embedding[int(i[0]),:], embedding[int(i[1]),:]))
        x = torch.vstack((x,row))
    return x[1:].requires_grad_(True) #To avoid the first row used to define the dimensions
    










        
        
        
        