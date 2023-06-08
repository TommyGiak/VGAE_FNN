# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:39:16 2023

@author: Tommaso Giacometti
"""
import torch
import models
import plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(5)

#Dataset selection
dataset = 'cora'

if dataset == 'bio':
    data = models.Data_Bio()
if dataset == 'cora':
    data = models.Data_Papers()
    
#VGAE
in_channels = data.x.shape[1]
hid_dim = 100
emb_dim = 50

autoencoder = models.VGAE(in_channels, hid_dim, emb_dim).to(device)

print(f'{plots.Bcolors.HEADER}Training of the VGAE{plots.Bcolors.ENDC}')
lossi_VGAE = autoencoder.train_cycle(data, epochs=1000)#Training VGAE


#Data processing for the FNN
embedding = autoencoder(data.x, data.train_pos)[0].detach() #[0] -> To get only z and not logvar
data_fnn = models.Data_FNN(embedding, data)

#FNN
fnn = models.FNN(emb_dim*2).to(device)

#Train
print(f'{plots.Bcolors.HEADER}Training of the FNN{plots.Bcolors.ENDC}')
lossi_fnn, lossi_test_fnn = fnn.train_cycle_fnn(data_fnn)


#Plots
#VGAE
plots.plot_loss(lossi_VGAE, mean = 5, tit='Loss of the VGAE')

plots.plot_train_distribution_VGAE(autoencoder, data)
plots.plot_test_distribution_VGAE(autoencoder, data)

#FNN
plots.plot_loss(lossi_fnn, tit = 'Loss of the FNN')

plots.plot_train_distribution_FNN(fnn, embedding, data_fnn, test = False)
plots.plot_train_distribution_FNN(fnn, embedding, data_fnn, test = True)

