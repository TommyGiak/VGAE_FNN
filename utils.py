# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:12:49 2023

@author: Tommaso Giacometti
"""

import torch
from torch import Tensor
import numpy as np
from plots import Bcolors


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