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
- [matplotlib](https://matplotlib.org/stable/) (quite common)
- [numpy](https://numpy.org/install/) (even more common)

## How to use
Move into the cloned folder from the terminal and run the main file:
```shell
python main.py
```
The dataset to use can be choosed in the main file. The default epochs for the VGAE and FNN training may not be enough and can be changed in the main file.

## Datasets 
Up to now I tried with some different dataset to understand the generalization of this model in different systems.\
The best results are obtained with the biological protein-protein interactions, while for the citation papers datasets and for the Twitch dataset the results are just decent: the predictions for the link are not to bad in general but there exist other models that can outperform this scripts.

## Best results (up to now)
I performed a 'long' training in [Google Colab](https://colab.research.google.com) with the GPU runtime using the HPRD (Human Protein Reference Database) dataset, which is also the biggest, for more details on this dataset look at the paper cited above.\
The training involved 100k epochs for the VGAE (which are not so useful) and 300k epochs for the FNN. The computational time took 495s for the VGAE and 825s for the FNN. According to the parameters used in the paper, the results are:
- accuracy: 0.9731
- sensitivity: 0.9729
- specificity: 0.9741
- precision: 0.9918
- f-score: 0.9822
