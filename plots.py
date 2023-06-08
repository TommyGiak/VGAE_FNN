# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:00:08 2023

@author: Tommaso Giacometti
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

class Bcolors:
    #Class to print on terminal with different colors
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_loss(lossi : list, mean = 5) -> None:
    '''
    Plot the loss history in log scale.
    
    Parameters
    ----------
    lossi : list, ArrayLike
    
    mean : Optional
        The plot will show the mean of the loss for this number of steps.

    Returns
    -------
    Show the plot
    '''
    try:
        lossi = np.array(lossi)
        y = lossi.reshape(-1,mean).mean(axis=1)
        x = np.linspace(1, len(y), num=len(y))
        fig, ax = plt.subplots()
        ax.plot(x,y)
        ax.set_title(f'Mean of {mean} losses steps')
        ax.set_ylabel('loss')
        ax.set_xlabel(f'epoch/{mean}')
        ax.set_yscale('log')
        plt.show()
        pass
    except:
        print(f'{Bcolors.WARNING}WARNING : {Bcolors.ENDC}the shape of lossi is not multiple of {mean}!')
        print('The loss track plot will not be shown')
        pass


def plot_train_distribution_VGAE(model, x : Tensor, edges : Tensor) -> None:
    '''
    Plot the distribution of link probability for the given edges, only for the TRAIN set.
    
    Parameters
    ----------
    model : VGAE (or GAE)
    x : Tensor
        Feature tensor NxF, where N is the number of nodes and F the features for each node.
    edges : Tensor
        Adjacency matrix in SPARSE format: 2xE, where E is the number of edges.

    Returns
    -------
    Show the histogram of the distribution
    '''
    model.eval()
    with torch.no_grad():
        z = model.encode(x,edges)
        out = model.decode(z, edges)
        out = out.cpu().numpy()
    fig, ax = plt.subplots()
    ax.hist(out, bins = 30, label = f'{edges.shape[1]} total edges')
    ax.legend()
    ax.set_xlabel('Probability of link')
    ax.set_ylabel('Number of edges')
    ax.set_title('Train edges')
    plt.show()
    pass


def plot_test_distribution_VGAE(model, x : Tensor, train_pos : Tensor, test_pos : Tensor, test_neg : Tensor = None) -> None:
    '''
    Plot the distribution of link probability for the given edges, only for the VAL/TEST set.
    The test negative edges can be included or not.
    
    Parameters
    ----------
    model : VGAE (or GAE)
    x : Tensor
        Feature tensor NxF, where N is the number of nodes and F the features for each node.
    edges : Tensor
        Adjacency matrix in SPARSE format: 2xE, where E is the number of edges.

    Returns
    -------
    Show the histogram of the distribution
    '''
    model.eval()
    with torch.no_grad():
        z = model.encode(x,train_pos)
        pos_out = model.decode(z, test_pos)
        pos_out = pos_out.cpu().numpy()
        
    if test_neg is None:
        fig, ax = plt.subplots()
        ax.hist(pos_out, bins = 30, label = f'{test_pos.shape[1]} total edges')
        ax.legend()
        ax.set_xlabel('Probability of link')
        ax.set_ylabel('Number of edges')
        ax.set_title('Positive test edges')
        plt.show()
    else:
        with torch.no_grad():
            neg_out = model.decode(z, test_neg)
            neg_out = neg_out.cpu().numpy()
        fig, (ax1,ax2) = plt.subplots(1,2, figsize = (8,4))
        ax1.hist(pos_out, bins = 30, label = f'{test_pos.shape[1]} total edges')
        ax1.legend()
        ax1.set_xlabel('Probability of link')
        ax1.set_ylabel('Number of edges')
        ax1.set_title('Positive test edges')
        ax2.hist(neg_out, bins = 30, label = f'{test_neg.shape[1]} total edges')
        ax2.legend()
        ax2.set_xlabel('Probability of link')
        ax2.set_ylabel('Number of edges')
        ax2.set_title('Negative test edges')
        plt.show()   
    pass

