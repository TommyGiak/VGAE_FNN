# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:21:48 2023

@author: Tommaso Giacometti
"""
import torch
from torch import Tensor
import torch_geometric as pyg
from torch import nn
from torch_geometric.utils import negative_sampling, train_test_split_edges


class DataProcessing():
    '''
    Class to group the data in a more intuitive way.
    ONLY for paper's citation datasets (not for bio-data)

    Parameters
    ----------
    data : Dataset -> CORA, CiteSeer, PubMed

    Returns
    -------
    Data class.
    '''
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
    '''
    Graph Convolutional Network for the VGAE -> one shared hidden layer and two separated output
    layers for the embedding variables mu and log_var.

    Parameters
    ----------
    in_channels : int
        Number of features in input: F where F is a dimension of the feature matrix X: NxF.
        
    hid_dim : int
        Dimension of the shared hidden layer.
        
    emb_dim : int
        Output dimension (which is the dimension of the embedding).
    '''
    def __init__(self, in_channels : int, hid_dim : int, emb_dim : int):
        super().__init__()
        self.conv = pyg.nn.GCNConv(in_channels, hid_dim, cached=True) #cached -> True for storing the computation of the normalized adj matrix
        self.mu = pyg.nn.GCNConv(hid_dim, emb_dim, cached=True)
        self.logvar = pyg.nn.GCNConv(hid_dim, emb_dim, cached=True)
        
        
    def forward(self, x : Tensor, edges : Tensor):
        x = self.conv(x,edges).relu()
        return self.mu(x,edges), self.logvar(x,edges)
    
    
    
class VGAE(pyg.nn.VGAE):
    '''
    Variational Graph AutoEncoder which contain the GCNEncoder.

    Parameters
    ----------
    in_channels : int
        Number of features in input: F where F is a dimension of the feature matrix X: NxF.
        
    hid_dim : int
        Dimension of the shared hidden layer.
        
    emb_dim : int
        Output dimension (which is the dimension of the embedding).
    '''
    def __init__(self, in_channels, hid_dim, emb_dim):
        super().__init__(encoder=GCNEncoder(in_channels, hid_dim, emb_dim))
    
   
    def train_step(self, data, optimizer) -> float:
        '''
        Perform a single step of the training of the VGAE.

        Parameters
        ----------
        data : Data structure in according with the DataProcessing's class
            
        optimizer : torch.optim.* 
            The choosen optimizer for the training.

        Returns
        -------
        loss : float
            Loss of the training stap.
        '''
        self.train()
        optimizer.zero_grad()
        norm = 1/data.x.shape[0]
        z = self.encode(data.x, data.train_pos)
        loss = self.recon_loss(z, data.train_pos, data.train_neg) + self.kl_loss()*norm
        loss.backward()
        optimizer.step()
        return float(loss)
    
    
    def test_step(self, data) -> None:
        '''
        Perform a test step of the VGAE

        Parameters
        ----------
        data : Data structure in according with the DataProcessing's 

        Returns
        -------
        None.

        '''
        self.eval()
        with torch.no_grad():
            z = self.encode(data.x, data.train_pos)
            auc, ap = self.test(z, data.test_pos, data.test_neg)
        print(f'AUC: {auc:.4f} | AP: {ap:.4f}')
        pass
 
    
    def train_cycle(self, data , epochs : int = 1000, optimizer = None) -> list:
        '''
        Perform a cycle of training using the train_step function.

        Parameters
        ----------
        data : Data structure in according with the DataProcessing's 
            
        epochs : int, optional
            Number of epochs. The default is 1000.
            
        optimizer : torch.optim.*, optional
            The choosen optimizer for the training. If None it takes as default Adam with lr=1e-3
            
        Returns
        -------
        lossi : list
            List of all the loss steps of the training.
        '''
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


class  FNN(nn.Module):
    '''
    Feedfarward Neural Network, by default it has always two hidden layers with 128 and 64 neurons.

    Parameters
    ----------
    inputs : int
        Number of inputs (it should be twice the embedding dimension).
    out : int, optional
        Number of outputs. The default is 2 (binary classification).
    '''
    def __init__(self, inputs : int, out : int = 2):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(), nn.Dropout(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(),
                                 nn.Linear(32, out))
        
        
    def forward(self,x):
        logits = self.seq(x)
        return logits
        
    
    
def get_ffnn_input(embedding : Tensor, links : Tensor) -> Tensor:
    '''
    Generate the input for the FNN.

    Parameters
    ----------
    embedding : Tensor
        The embedding of the VGAE of dimension NxD, where N is the number of nodes and D the embedding dimension.
    links : Tensor (Sparse)
        Adjacency sparse matrix of the links to transform of dimension 2xL.

    Returns
    -------
    Tensor
        Inputs for the FNN of dimension Lx2D.

    '''
    x = torch.hstack((embedding[0,:],embedding[1,:]))
    for i in links.T:
        row = torch.hstack((embedding[int(i[0]),:], embedding[int(i[1]),:]))
        x = torch.vstack((x,row))
    return x[1:].requires_grad_(True) #To avoid the first row used to define the dimensions
    










        
        
        
        