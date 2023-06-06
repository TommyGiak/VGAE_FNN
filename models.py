# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:21:48 2023

@author: Tommaso Giacometti
"""
import torch
import torch_geometric as pyg
from torch import nn
from torch_geometric.utils import negative_sampling, train_test_split_edges


class DataProcessing():
    def __init__(self, data):
        self.x = data.x
        self.all_index = data.edge_index
        data = train_test_split_edges(data, 0.05, 0.1)
        self.train_pos = data.train_pos_edge_index
        self.train_neg = negative_sampling(self.all_index,
                                           num_nodes= data.x.shape[0], 
                                           num_neg_samples=self.train_pos.shape[1])
        self.val_pos = data.val_pos_edge_index
        self.val_neg = data.val_neg_edge_index
        self.test_pos = data.test_pos_edge_index
        self.test_neg = data.test_neg_edge_index
        del self.all_index
        

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
    
   
    def train_step(self, data, optimizer):
        self.train()
        optimizer.zero_grad()
        norm = 1/data.x.shape[0]
        z = self.encode(data.x, data.train_pos)
        loss = self.recon_loss(z, data.train_pos, data.train_neg) + self.kl_loss()*norm
        loss.backward()
        optimizer.step()
        return float(loss)
    
    
    def test_step(self, data):
        self.eval()
        with torch.no_grad():
            z = self.encode(data.x, data.train_pos)
            auc, ap = self.test(z, data.test_pos, data.test_neg)
        print(f'AUC: {auc:.4f} | AP: {ap:.4f}')
        pass
 
    
    def train_cycle(self, data , epochs = 1000, optimizer = None):
        lossi = []
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(),lr=1e-2)
        for i in range(epochs):
            lossi.append(self.train_step(data, optimizer=optimizer))
            if i%(epochs/20) == 0:
                print(f'{i/epochs*100:.2f}% | loss = {lossi[i]:.4f} -> ', end = '')
                self.test_step(data)
        print(f'100.00% | loss = {lossi[i]:.4f} -> ', end = '')
        self.test_step(data)
        return lossi


class  FFNN(nn.Module):
    
    def __init__(self, inputs, out = 2):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(), nn.Dropout(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(),
                                 nn.Linear(32, out))
        
        
    def forward(self,x):
        logits = self.seq(x)
        return logits
        
    
def get_ffnn_input(embedding, links):
    x = torch.hstack((embedding[0,:],embedding[1,:]))
    for i in links.T:
        row = torch.hstack((embedding[int(i[0]),:], embedding[int(i[1]),:]))
        x = torch.vstack((x,row))
    return x[1:].requires_grad_(True) #To avoid the first row used to define the dimensions
    










        
        
        
        