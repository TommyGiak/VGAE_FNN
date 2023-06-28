# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:12:49 2023

@author: Tommaso Giacometti
"""

import torch
import torch_geometric as pyg
from torch import Tensor
import numpy as np
from plots import Bcolors
from torch_sparse import spspmm, coalesce
import matplotlib.pyplot as plt



EPS = 1e-15


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
    return x.requires_grad_(True)
    

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
    out = dict(accuracy=accuracy, sensitivity=sensitivity, specificity=specificity, precision=precision,fscore=fscore)         
    return out


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
    out = dict(accuracy=accuracy, sensitivity=sensitivity, specificity=specificity, precision=precision,fscore=fscore)         
    return out
    
        
def column_norm(data : Tensor) -> Tensor:
    features = torch.tensor([data.shape[1]])
    mean = torch.mean(data, 0)
    var = torch.var(data, dim=0)
    std = torch.sqrt(var)
    normal = (data - mean)/(std+EPS)
    return normal/torch.sqrt(features)


def print_dict(dictionary : dict, part = None) -> None:
    print()
    if part is not None:
        print(part)
    for e in dictionary:
        print(f'{e}: {dictionary[e]:.5f}')
    pass


def adj_with_neighbors(adj : Tensor, mat_dim : int, order : int, plot : bool = True):
    values = torch.ones_like(adj[0],requires_grad=False).float()
    adj.requires_grad_(False)
    
    n_pow, val_pow = adj, values
    n_neig = [n_pow.shape[1]]
    i=0
    
    tot_adj = adj.clone()
    tot_val = values.clone()
    
    for i in range(order-1):
        n_pow, val_pow = spspmm(n_pow, val_pow, adj, values, mat_dim, mat_dim, mat_dim)
        n_neig.append(n_pow.shape[1])
        n_pow, val_pow = pyg.utils.remove_self_loops(n_pow,val_pow)
        
        # transform_val = (val_pow+1).clone().log10() # old implementation
        # transform_val = torch.pow(0.8*transform_val/torch.max(transform_val), i+2)
        mean = val_pow.mean()
        std = val_pow.std()
        transform_val = (val_pow.clone()-mean)/std
        transform_val = torch.sigmoid(transform_val)
        transform_val = torch.pow(transform_val, i+1)
        # transform_val = transform_val/(i+1)

        tot_adj = torch.hstack((tot_adj,n_pow))
        tot_val = torch.hstack((tot_val,transform_val))
        if len(val_pow) == 0:
            break        
                
    if plot:
        exp = list(range(i+2))
        exp = [i+1 for i in exp]
        fig, ax = plt.subplots()
        ax.bar(exp, n_neig)
        ax.set_title('Total number of link for order of the adj matrix')
        ax.set_xlabel('adj matrix order')
        ax.set_ylabel('Number of links')
        plt.show()
        
    tot_adj, tot_val = coalesce(tot_adj, tot_val, mat_dim, mat_dim)
    

    return tot_adj, tot_val

def reconstruct_graph(data, data_fnn, model):
    model.eval()
    with torch.no_grad():
        out_pos = model(data_fnn.test_emb_pos)
        out_neg = model(data_fnn.test_emb_neg)
        links = data.train_pos.clone()
        
    for i, out in enumerate(out_pos):
        if torch.argmax(out) == 1:
            links = torch.hstack((links,data.test_pos[:,i:i+1]))
    for i, out in enumerate(out_neg):
        if torch.argmax(out) == 1:
            links = torch.hstack((links,data.test_neg[:,i:i+1]))            
    return links
                    
    
    
    
    
    
    
    
    
    