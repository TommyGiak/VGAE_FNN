#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:00:08 2023

@author: Tommaso Giacometti
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

class bcolors:
    #Class to print on terminal in different colors
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

    Parameters
    ----------
    lossi : list
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    '''
    try:
        lossi = np.array(lossi)
        y = lossi.reshape(-1,mean).mean(axis=1)
        x = np.linspace(1, len(y), num=len(y))
        fig, ax = plt.subplots()
        ax.plot(x,y)
        ax.set_title(f'mean of {mean} losses steps')
        ax.set_ylabel('loss')
        ax.set_xlabel(f'epoch/{mean}')
        ax.set_yscale('log')
        plt.show()
        pass
    except:
        print(f'{bcolors.WARNING}WARNING : {bcolors.ENDC}the shape of lossi is not multiple of {mean}!')
        print('The loss track plot will not be shown')
        pass


def plot_train_distribution(model, x, edges) -> None:
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


def plot_test_distribution(model, x, train_pos, test_pos, test_neg = None) -> None:
    
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

