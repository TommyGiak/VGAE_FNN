# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:39:16 2023

@author: Tommaso Giacometti
"""
from time import time
import torch
import models
import utils
import plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)

#Dataset selection
dataset = 'coli'
order = 4

if dataset == 'coli':
    data = models.Data_Bio_Coli(order=order)
elif dataset == 'cora' or dataset == 'pubmed' or dataset == 'citeseer':
    data = models.Data_Papers(dataset, order=order)
elif dataset == 'human':
    data = models.Data_Bio_Human(order=order)
elif dataset == 'twitch':
    data = models.Data_Twitch(order=order)
elif dataset == 'fake':
    data = models.Data_Fake(order=order)
else:
    raise ValueError(f'There not exist {dataset} dataset')    

#%%
#VGAE
in_channels = data.x.shape[1]
hid_dim = 100
emb_dim = 50

autoencoder = models.VGAE(in_channels, hid_dim, emb_dim).to(device)

start_vgae = time()
print(f'{plots.Bcolors.HEADER}Training of the VGAE{plots.Bcolors.ENDC}')
lossi_VGAE = autoencoder.train_cycle(data, weights=True, epochs=5000, include_neg=True)#Training VGAE
stop_vgae = time()

#Data processing for the FNN
embedding = autoencoder(data.x, data.train_pos)[0].detach() #[0] -> To get only z and not logvar
data_fnn = models.Data_FNN(embedding, data)

#FNN
fnn = models.FNN(emb_dim*2).to(device)

#Train
start_fnn = time()
print(f'{plots.Bcolors.HEADER}Training of the FNN{plots.Bcolors.ENDC}')
lossi_fnn, lossi_test_fnn = fnn.train_cycle_fnn(data_fnn, epochs=20000)
stop_fnn = time()

#Computational times
print(f'The training of the VGAE took {stop_vgae-start_vgae} sec')
print(f'The training of the FNN took {stop_fnn-start_fnn} sec')

#Plots
#VGAE
plots.plot_loss(lossi_VGAE, mean = 5, tit='Loss of the VGAE')

plots.plot_train_distribution_VGAE(autoencoder, data)
plots.plot_test_distribution_VGAE(autoencoder, data)

#FNN
plots.plot_loss(lossi_fnn, tit = 'Loss of the FNN', mean = 500)

plots.plot_distribution_FNN(fnn, embedding, data_fnn, test = False)
plots.plot_distribution_FNN(fnn, embedding, data_fnn, test = True)

#Results
vgae_results = utils.get_argmax_VGAE(autoencoder, data)
fnn_results = utils.get_argmax_FNN(fnn, data_fnn)

utils.print_dict(vgae_results, part = 'VGAE results in the classification')
utils.print_dict(fnn_results, part = 'FNN result in the classification')



