# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:21:48 2023

@author: Tommaso Giacometti
"""
import torch
from torch import Tensor
import torch_geometric as pyg
from torch_geometric.datasets import Planetoid, Twitch
from torch import nn
from torch_geometric.utils import negative_sampling, train_test_split_edges
from plots import Bcolors
import os
import numpy as np

EPS = 1e-15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Data_Papers():
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
    def __init__(self, name : str):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = current_dir + '/data'
        
        # norm = pyg.transforms.NormalizeFeatures()
        # dataset = Planetoid(root=data_dir, name='Cora', transform=norm) # Old (and may wrong) implementation of the normalization 
        if name == 'cora':
            dataset = Planetoid(root=data_dir, name='Cora')
        elif name == 'pubmed':
            dataset = Planetoid(root=data_dir, name='PubMed')
        elif name == 'citeseer':
            dataset = Planetoid(root=data_dir, name='CiteSeer')
        data = dataset[0].to(device)
        
        # self.x = data.x # Still the old normalization
        self.x = column_norm(data.x)
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
        del self.all_index, data, dataset
        

class Data_Twitch():
    def __init__(self, name : str = 'ES'):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = current_dir + '/data'
        
        # norm = pyg.transforms.NormalizeFeatures()
        # dataset = Planetoid(root=data_dir, name='Cora', transform=norm) # Old (and may wrong) implementation of the normalization 
        dataset = Twitch(root=data_dir, name=name)
        data = dataset[0].to(device)
                
        # self.x = data.x # Still the old normalization
        self.x = column_norm(data.x)
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
        del self.all_index, data, dataset
        
    
class Data_Bio_Coli():
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = current_dir + '/data'
        breakpoint()
        features = np.load(data_dir + '/Coli_features.npz')['Coli_features'].astype(np.float32)
        pos_edges = np.load(data_dir + '/Coli_PositiveEdges.npz')['Coli_PositiveEdges'].astype(np.int64)
        neg_edges = np.load(data_dir + '/Coli_NegativeEdges.npz')['Coli_NegativeEdges'].astype(np.int64)

        features = torch.from_numpy(features).to(device)
        self.x = column_norm(features)
        # soft = torch.nn.Softmax(dim = 1) # Old (and may wrong) implementation of the normalization
        # self.x = soft(features)
        # del soft
        del features

        
        pos_edges = torch.from_numpy(pos_edges).to(device)
        neg_edges = torch.from_numpy(neg_edges).to(device)

        idx_pos = torch.randperm(pos_edges.shape[0])
        idx_neg = torch.randperm(neg_edges.shape[0])
        pos_edges = pos_edges[idx_pos].t()
        neg_edges = neg_edges[idx_neg].t()

        ind_pos = int(pos_edges.shape[1]*0.85)
        ind_neg = int(neg_edges.shape[1]*0.85)

        self.train_pos = pos_edges[:,:ind_pos]
        self.train_neg = neg_edges[:,:ind_neg]
        self.test_pos = pos_edges[:,ind_pos:]
        self.test_neg = neg_edges[:,ind_neg:]
        del pos_edges, neg_edges, idx_pos, idx_neg, ind_pos, ind_neg
        
        
class Data_Bio_Human():
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = current_dir + '/data'
        features = np.load(data_dir + '/H_features.npz')['features'].astype(np.float32)
        pos_edges = np.load(data_dir + '/H_PositiveEdges.npz')['H_PositiveEdges'].astype(np.int64)
        neg_edges = np.load(data_dir + '/H_NegativeEdges.npz')['H_NegativeEdges'].astype(np.int64)

        features = torch.from_numpy(features).to(device)
        self.x = column_norm(features)
        # soft = torch.nn.Softmax(dim = 1) # Old (and may wrong) implementation of the normalization
        # self.x = soft(features)
        # del soft
        del features
        
        pos_edges = torch.from_numpy(pos_edges).to(device)
        neg_edges = torch.from_numpy(neg_edges).to(device)

        idx_pos = torch.randperm(pos_edges.shape[0])
        idx_neg = torch.randperm(neg_edges.shape[0])
        pos_edges = pos_edges[idx_pos].t()
        neg_edges = neg_edges[idx_neg].t()

        ind_pos = int(pos_edges.shape[1]*0.85)
        ind_neg = int(neg_edges.shape[1]*0.85)

        self.train_pos = pos_edges[:,:ind_pos]
        self.train_neg = neg_edges[:,:ind_neg]
        self.test_pos = pos_edges[:,ind_pos:]
        self.test_neg = neg_edges[:,ind_neg:]
        del pos_edges, neg_edges, idx_pos, idx_neg, ind_pos, ind_neg
        
        
class Data_FNN():
    def __init__(self, embedding, data):
        self.train_emb_pos = get_fnn_input(embedding, data.train_pos)
        self.train_emb_neg = get_fnn_input(embedding, data.train_neg)
        self.test_emb_pos = get_fnn_input(embedding, data.test_pos)
        self.test_emb_neg = get_fnn_input(embedding, data.test_neg)
        
        

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
    
   
    def train_step(self, data, optimizer, include_neg : bool = True) -> float:
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
        if include_neg:
            loss = self.recon_loss(z, data.train_pos, data.train_neg) + self.kl_loss()*norm
        elif not include_neg:
            loss = self.recon_loss(z, data.train_pos) + self.kl_loss()*norm
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
 
    
    def train_cycle(self, data , epochs : int = 1000, optimizer = None, include_neg : bool = True) -> list:
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
            lossi.append(self.train_step(data, optimizer=optimizer, include_neg=include_neg))
            if i%(epochs/20) == 0:
                print(f'{i/epochs*100:.2f}% | loss = {lossi[i]:.4f} -> ', end = '')
                self.test_step(data)
        print(f'100.00% | loss = {lossi[i]:.4f} -> ', end = '')
        self.test_step(data)
        return lossi


class  FNN(nn.Module):
    '''
    Feedfarward Neural Network, by default it has always three hidden layers with 128, 64 and 32 neurons.

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
    
    
    def train_fnn(self, pos, neg, optim, loss_fn):
        self.train()
        one = torch.ones_like(pos[:,0], dtype=torch.long)
        zero = torch.zeros_like(neg[:,0], dtype=torch.long)
        optim.zero_grad()
        out_pos = self(pos)
        out_neg = self(neg)
        loss = loss_fn(out_pos, one) + loss_fn(out_neg, zero)
        loss.backward()
        optim.step()
        return loss.item()


    def test_fnn(self, test_emb_pos, test_emb_neg, loss_fn = None):
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        self.eval()
        with torch.no_grad():
            one = torch.ones_like(test_emb_pos[:,0], dtype=torch.long)
            zero = torch.zeros_like(test_emb_neg[:,0], dtype=torch.long)
            out_pos = self(test_emb_pos)
            out_neg = self(test_emb_neg)
            loss = loss_fn(out_pos, one) + loss_fn(out_neg, zero)
        return loss.item()
        
    
    def train_cycle_fnn(self, data, batch_size = 128, epochs = 10000, optim = None, loss_fn = None):
        if optim is None:
            optim = torch.optim.Adam(self.parameters(), lr = 1e-03)
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        lossi = []
        lossi_test = []
        index_pos = np.random.randint(0, data.train_emb_pos.shape[0]-batch_size, epochs)
        index_neg = np.random.randint(0, data.train_emb_neg.shape[0]-batch_size, epochs)
        for i in range(epochs):
            lossi.append(self.train_fnn(data.train_emb_pos[index_pos[i]:index_pos[i]+batch_size], 
                                        data.train_emb_neg[index_neg[i]:index_neg[i]+batch_size], 
                                        optim, loss_fn))
            if i%(epochs/50) == 0:
                lossi_test.append(self.test_fnn(data.test_emb_pos, data.test_emb_neg))
                print(f'{i/epochs * 100:.2f}% -> Train loss: {lossi[-1]:.3f} | Test loss: {lossi_test[-1]:.3f}')
        lossi_test.append(self.test_fnn(data.test_emb_pos, data.test_emb_neg))
        print(f'100.00% -> Train loss: {lossi[-1]:.4f} | Test loss: {lossi_test[-1]:.4f}')
        return lossi, lossi_test
    

def get_fnn_input(embedding : Tensor, links : Tensor) -> Tensor:
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
    col1 = torch.squeeze(embedding[links.T[:,0:1]])
    col2 = torch.squeeze(embedding[links.T[:,1:2]])
    x = torch.hstack((col1,col2))
    return x.requires_grad_(True) #To avoid the first row used to define the dimensions
    

def minibatch(tens : Tensor, batch_size : int = 32, shuffle : bool = True) -> Tensor:
    '''
    Transform the tensor in a iterable tensor divided in batch_size's tensors.
    WRANING : if the number of rows of tens is not a multiple of batch_size, a part of the samples will be wasted.

    Parameters
    ----------
    tens : Tensor
        Tensor of dimension *x2D to be dividen in batches.
    batch_size : int, optional
        The default is 32.
    shuffle : bool, optional
        Shuffle the row of the input tensor. The default is True.

    Returns
    -------
    tens : Tensor
        tensor of size: batches x batch_size x 2D.

    '''
    if shuffle:
        ind = torch.randperm(tens.shape[0])
        tens = tens[ind]
        batches = tens.shape[0]//batch_size
    if tens.shape[0]%batch_size == 0:
        tens = tens.view(batches, batch_size, tens.shape[1])
    else:
        print(f'{Bcolors.WARNING}WARNING : {Bcolors.ENDC} since the number of ', end='')
        print(f'training sample {tens.shape[0]} is not a multiple of {batch_size}, ', end='')
        print(f'{tens.shape[0]%batch_size} random samples will be wasted.')
        tens = tens[:batches*batch_size]
        tens = tens.view(batches, batch_size, tens.shape[1])
    return tens


def get_argmax_VGAE(model, data):
    tp = 0
    fp = 0
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos)
        pos_out = model.decode(z, data.test_pos)
        pos_out = pos_out.cpu().numpy()
        neg_out = model.decode(z, data.test_neg)
        neg_out = neg_out.cpu().numpy()
    for p in pos_out:
        if p > 0.5:
            tp += 1
    for p in neg_out:
        if p > 0.5:
            fp += 1
    tn = len(neg_out) - fp
    fn = len(pos_out) - tp
    assert tp + fn == len(pos_out)
    assert fp + tn == len(neg_out)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    fscore = 2*precision*sensitivity/(precision+sensitivity)            
    return accuracy, sensitivity, specificity, precision, fscore


def get_argmax_FNN(model, data):
    model.eval()
    with torch.no_grad():
        h_pos = torch.nn.functional.softmax(model(data.test_emb_pos), dim = 1)
        h_neg = torch.nn.functional.softmax(model(data.test_emb_neg), dim = 1)
        h_pos = torch.argmax(h_pos, dim = 1)
        h_neg = torch.argmax(h_neg, dim = 1)
        h_pos = h_pos.detach().cpu().numpy()
        h_neg = h_neg.detach().cpu().numpy()
    tp = np.sum(h_pos)
    fp = np.sum(h_neg)
    tn = len(h_neg) - fp
    fn = len(h_pos) - tp
    assert tp + fn == len(h_pos)
    assert fp + tn == len(h_neg)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    fscore = 2*precision*sensitivity/(precision+sensitivity)            
    return accuracy, sensitivity, specificity, precision, fscore
    
        
def column_norm(data : Tensor) -> Tensor:
    mean = torch.mean(data, 0)
    var = torch.var(data, dim=0)
    std = torch.sqrt(var)
    normal = (data - mean)/(std+EPS)
    return normal/torch.max(normal)
    
    
    
    
    
        