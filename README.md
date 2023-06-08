# VGAE-FNN for link prediction
## Introduction
This repository contains an implementation of a VGAE (variational graph auto-encoder) connected to a FNN (feedforward neural network) for link prediction.\
This work is inspired by the model presented in [this paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03646-8). Actually the first goal is to reproduce the results presented in the paper and then try to study the validity of the model with different types of datasets.
Up to now there's a massive lack of documentation that will be added soon, hopefully.\
This model is implemented in [PyTorch](https://pytorch.org) with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#).

## Install
From terminal go into an empty folder and clone this repository:
```shell
git clone https://github.com/TommyGiak/VGAE_FNN.git
```

## Requirements 
- [python](https://www.python.org)
- [pytorch](https://pytorch.org/get-started/locally/)
- [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#)
- [matplotlib](https://matplotlib.org/stable/)
- [numpy](https://numpy.org/install/)

## How to use
Move into the cloned folder from the terminal and run the main file:
```shell
python main.py
```
