# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:00:08 2023

@author: Tommaso Giacometti
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def plot_loss(lossi : list, mean = 20, tit = None) -> None:
    '''
    Plot the loss history in log scale.
    
    Parameters
    ----------
    lossi : list, ArrayLike
    
    mean : Optional
        The plot will show the mean of the loss for this number of steps.
        
    tit : str, optional
        Title to put on the plot.

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
        if tit is None:
            ax.set_title(f'Mean of {mean} losses steps')
        else:
            ax.set_title(tit)
        ax.set_ylabel('loss')
        ax.set_xlabel(f'epoch/{mean}')
        ax.set_yscale('log')
        plt.show()
        pass
    except:
        print(f'{Bcolors.WARNING}WARNING : {Bcolors.ENDC}the shape of lossi is not multiple of {mean}!')
        print('The loss track plot will not be shown')
        pass


def plot_train_distribution_VGAE(model, data) -> None:
    '''
    Plot the distribution of link probability for the given edges, only for the TRAIN set.
    
    Parameters
    ----------
    model : VGAE (or GAE)
    data : Data class according to Data_Paper or Data_Bio

    Returns
    -------
    Show the histogram of the distribution
    '''
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos)
        pos_out = model.decode(z, data.train_pos)
        pos_out = pos_out.cpu().numpy()
        neg_out = model.decode(z, data.train_neg)
        neg_out = neg_out.cpu().numpy()
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (9,4))
    fig.suptitle('VGAE distributions')
    ax1.hist(pos_out, bins = 30, label = f'{data.train_pos.shape[1]} total edges')
    ax1.legend()
    ax1.set_xlabel('Probability of link')
    ax1.set_ylabel('Number of edges')
    ax1.set_title('Positive train edges')
    ax2.hist(neg_out, bins = 30, label = f'{data.test_neg.shape[1]} total edges')
    ax2.legend()
    ax2.set_xlabel('Probability of link')
    ax2.set_title('Negative train edges')
    plt.show()   
    pass


def plot_test_distribution_VGAE(model, data) -> None:
    '''
    Plot the distribution of link probability for the given edges, only for the VAL/TEST set.
    
    Parameters
    ----------
    model : VGAE (or GAE)
    data : Data class according to Data_Paper or Data_Bio


    Returns
    -------
    Show the histogram of the distribution
    '''
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos)
        pos_out = model.decode(z, data.test_pos)
        pos_out = pos_out.cpu().numpy()
        neg_out = model.decode(z, data.test_neg)
        neg_out = neg_out.cpu().numpy()
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (9,4))
    fig.suptitle('VGAE distributions')
    ax1.hist(pos_out, bins = 30, label = f'{data.test_pos.shape[1]} total edges')
    ax1.legend()
    ax1.set_xlabel('Probability of link')
    ax1.set_ylabel('Number of edges')
    ax1.set_title('Positive test edges')
    ax2.hist(neg_out, bins = 30, label = f'{data.test_neg.shape[1]} total edges')
    ax2.legend()
    ax2.set_xlabel('Probability of link')
    ax2.set_title('Negative test edges')
    plt.show()   
    pass


def plot_distribution_FNN(model, embedding, data, test : bool) -> None:
    model.eval()
    with torch.no_grad():
        if test:
            h_pos = torch.nn.functional.softmax(model(data.test_emb_pos), dim = 1)
            h_neg = torch.nn.functional.softmax(model(data.test_emb_neg), dim = 1)
            h_pos = h_pos.detach().cpu().numpy()[:,1]
            h_neg = h_neg.detach().cpu().numpy()[:,1]
            tit = 'test'
        else:
            h_pos = torch.nn.functional.softmax(model(data.train_emb_pos), dim = 1)
            h_neg = torch.nn.functional.softmax(model(data.train_emb_neg), dim = 1)
            h_pos = h_pos.detach().cpu().numpy()[:,1]
            h_neg = h_neg.detach().cpu().numpy()[:,1]
            tit = 'train'
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (9,4))
    fig.suptitle('FNN distributions, probability for the link to exist')
    ax1.hist(h_pos, bins = 30, label = f'{len(h_pos)} total links')
    ax1.legend()
    ax1.set_xlabel('Probability of link')
    ax1.set_ylabel('Number of edges')
    ax1.set_title(f'Positive {tit} edges')
    ax2.hist(h_neg, bins = 30, label = f'{len(h_neg)} total links')
    ax2.legend()
    ax2.set_xlabel('Probability of link')
    ax2.set_title(f'Negative {tit} edges')
    plt.show()   
    pass




