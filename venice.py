# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:14:36 2023

@author: Tommaso Giacometti
"""
import numpy as np
import torch
import torch_geometric as pyg
import models
import networkx as nx
import utils
from time import time
import plots
import matplotlib.pyplot as plt

name = 'venice'

if name == 'venice':
    path = '/Users/tommygiak/Desktop/VGAE_FNN/data/cities/venice_edge_info.txt'
    n_nodes = 1840
elif name == 'bologna':
    path = '/Users/tommygiak/Desktop/VGAE_FNN/data/cities/bologna_edge_info.txt'
    n_nodes = 541    

data = np.loadtxt(path,dtype=np.float32)

edge_number = data[:,0].astype(np.int64)

#Adj matrix
link1 = data[:,1]
link2 = data[:,4]
edge_index = np.vstack((link1,link2)).astype(np.int64)
del link1, link2

#Node features
pos1 = data[:,2:4]
pos2 = data[:,5:7]
indexes = edge_index.transpose()
tot = np.hstack((indexes,pos1,pos2))
a = []
for j in range(n_nodes):
    i = j
    if i in data[:,1]:
        a.append([i, data[data[:,1]==i,2][0], data[data[:,1]==i,3][0]])
    elif i in data[:,4]:
        a.append([i, data[data[:,4]==i,5][0], data[data[:,4]==i,6][0]])
    else:
        raise IndexError(i)
pos = np.array(a)[:,1:].astype(np.float32)
pos_dict = {tuple(a):i for i,a in enumerate(pos)}
pos_dict = {i:a for a,i in pos_dict.items()}

edge_index = torch.from_numpy(edge_index)
pos = torch.from_numpy(pos)
edge_index = pyg.utils.to_undirected(edge_index,num_nodes=n_nodes)

data = pyg.data.Data(x=pos, edge_index=edge_index)

print(f'Number of nodes: {data.num_nodes}')

print(f'Number of edges: {data.num_edges}')

print(f'Has isolated nodes: {data.has_isolated_nodes()}')  # False
print(f'Has self-loops: {data.has_self_loops()}')  # False
print(f'Is undirected: {data.is_undirected()}')  # True

print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')

#%%

G = pyg.utils.to_networkx(data, to_undirected=True)

fig, ax = plt.subplots(dpi=1000) 
centrality = nx.betweenness_centrality(G, endpoints=True)
node_size = [v * 20 for v in centrality.values()]
nx.draw_networkx_nodes(G, pos_dict, node_size = node_size, node_shape='o', alpha=0.4)
nx.draw_networkx_edges(G, pos_dict, width = 0.5, edge_color="gainsboro")

ax.set_title('Venice street rapresentation')
plt.savefig('venice.pdf')
plt.show()

#%%

degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)

fig,(ax1,ax2) = plt.subplots(1,2, figsize=(8, 4))
fig.suptitle("Degree of a random graph")

ax1.plot(degree_sequence, "b-", marker="o")
ax1.set_title("Degree Rank Plot")
ax1.set_ylabel("Degree")
ax1.set_xlabel("Rank")

ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("# of Nodes")

fig.tight_layout()
plt.show()

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pyg.data.Data(x=pos, edge_index=edge_index)

data_model = models.Data_Venice(data, order = 3)

#VGAE
in_channels = data_model.x.shape[1]
hid_dim = 100
emb_dim = 50

autoencoder = models.VGAE(in_channels, hid_dim, emb_dim).to(device)

start_vgae = time()
print(f'{plots.Bcolors.HEADER}Training of the VGAE{plots.Bcolors.ENDC}')
lossi_VGAE = autoencoder.train_cycle(data_model, weights=False, epochs=5000, include_neg=False)#Training VGAE
stop_vgae = time()

#Data processing for the FNN
embedding = autoencoder(data_model.x, data_model.train_pos)[0].detach() #[0] -> To get only z and not logvar
data_model_fnn = models.Data_FNN(embedding, data_model)

#FNN
fnn = models.FNN(emb_dim*2).to(device)

#Train
start_fnn = time()
print(f'{plots.Bcolors.HEADER}Training of the FNN{plots.Bcolors.ENDC}')
lossi_fnn, lossi_test_fnn = fnn.train_cycle_fnn(data_model_fnn, epochs=20000)
stop_fnn = time()

#Computational times
print(f'The training of the VGAE took {stop_vgae-start_vgae} sec')
print(f'The training of the FNN took {stop_fnn-start_fnn} sec')

#Plots
#VGAE
plots.plot_loss(lossi_VGAE, mean = 5, tit='Loss of the VGAE')

plots.plot_train_distribution_VGAE(autoencoder, data_model)
plots.plot_test_distribution_VGAE(autoencoder, data_model)

#FNN
plots.plot_loss(lossi_fnn, tit = 'Loss of the FNN', mean = 500)

plots.plot_distribution_FNN(fnn, embedding, data_model_fnn, test = False)
plots.plot_distribution_FNN(fnn, embedding, data_model_fnn, test = True)

#Results
vgae_results = utils.get_argmax_VGAE(autoencoder, data_model)
fnn_results = utils.get_argmax_FNN(fnn, data_model_fnn)

utils.print_dict(vgae_results, part = 'VGAE results in the classification')
utils.print_dict(fnn_results, part = 'FNN result in the classification')


#%%

only_train_data = pyg.data.Data(x = data_model.x, edge_index=data_model.train_pos)

G1 = pyg.utils.to_networkx(only_train_data, to_undirected=True)

fig, ax = plt.subplots(dpi=1000) 
centrality = nx.betweenness_centrality(G1, endpoints=True)
node_size = [v * 20 for v in centrality.values()]
nx.draw_networkx_nodes(G1, pos_dict, node_size = node_size, node_shape='o', alpha=0.4)
nx.draw_networkx_edges(G1, pos_dict, width = 0.5, edge_color="gainsboro")

ax.set_title('Venice street rapresentation with missing links')
plt.savefig('venice_missing.pdf')
plt.show()


#%%
#Reconstructed graph

link_recon = utils.reconstruct_graph(data_model, data_model_fnn, fnn)

G_recon = pyg.data.Data(x=data.x, edge_index=link_recon)
G_recon = pyg.utils.to_networkx(G_recon, to_undirected=True)

fig, ax = plt.subplots(dpi=1000) 
centrality = nx.betweenness_centrality(G_recon, endpoints=True)
node_size = [v * 20 for v in centrality.values()]
nx.draw_networkx_nodes(G_recon, pos_dict, node_size = node_size, node_shape='o', alpha=0.4)
nx.draw_networkx_edges(G_recon, pos_dict, width = 0.5, edge_color="gainsboro")

ax.set_title('Venice street rapresentation (reconstruction)')
plt.savefig('venice_recon.pdf')
plt.show()

