import torch
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import os


def make_toy_data(num_examples, inpt_func, outpt_func):
    generator =  torch.rand(num_examples)
    inpt = inpt_func(generator)
    outpt = outpt_func(inpt)
    return inpt, outpt

def check_outpt(label, outpt, tol_neg, tol_pos):
    diff = outpt-label
    neg_acc = diff > tol_neg[None,None,:]
    pos_acc = diff < tol_pos[None,None,:]
    acc = neg_acc*pos_acc
    acc = acc.reshape(diff.size(0), -1)
    return torch.all(acc, dim=1)

def make_tol(std_dev, dim, add=1e-3, device = 'cpu'):
    tol_neg = - (torch.abs(torch.normal(mean=torch.zeros(dim), std=std_dev*torch.ones(dim))) + add).to(device)
    tol_pos = + (torch.abs(torch.normal(mean=torch.zeros(dim), std=std_dev*torch.ones(dim))) + add).to(device)
    return tol_neg, tol_pos

def make_func(dim_in,dim_out, seq_len):
    seq_add = torch.arange(0,1,1/seq_len)
    '''    di_a = [[i for i in range(dim_in)] for j in range(dim_in)]
    print('asdsd')
    index_ = list(itertools.product(*di_a))
    print(len(index_))
    ri = random.choices(index_, k=dim_out)
    tri = torch.tensor(ri)'''
    tri = torch.randint(low=0, high=dim_in, size=(dim_out, dim_in))
    signs = torch.randint(low=-1, high=2, size=(dim_in*dim_out,)).reshape(dim_out, dim_in)
    #print(signs)

    tri = tri.reshape(-1).repeat(seq_len).reshape(1,-1)
    seq_counter = torch.arange(0, seq_len).reshape(1,-1).repeat([dim_in*dim_out, 1]).T.reshape(1,-1)
    tri = torch.cat((seq_counter, tri), dim=0)
    def inner_fc(inpt):
        n = len(inpt)
        tri_n = tri.repeat([1, n])
        n_counter = torch.arange(0, n).reshape(1,-1).repeat([dim_in*dim_out*seq_len, 1]).T.reshape(1,-1)
        tri_n = torch.cat((n_counter, tri_n), dim=0)
        seq_inpt = inpt[:,None, :] + seq_add[None,:,None]
        result = seq_inpt[tuple(tri_n)]
        result = result.reshape(n,seq_len, dim_out, dim_in)
        result = result*signs[None,None,:,:]
        result[(signs==0)[None,None,:,:].repeat([result.size(0), result.size(1),1,1])]=1
        result = torch.prod(result, dim=-1)
        return result
    return inner_fc

def plot_fcts(inpt, neg_tol, pos_tol):
    inpt_dim = len(inpt[0])
    neg_inpt = (inpt + neg_tol[None,:]).numpy()
    pos_inpt = (inpt + pos_tol[None,:]).numpy()
    seq_len = len(inpt)
    np_inpt = inpt.numpy()
    num_graphs = int(np.ceil(np.sqrt(inpt_dim)))
    fig, ax = plt.subplots(num_graphs,num_graphs)
    fig.set_size_inches(9, 9)
    for sp in range(inpt_dim):
        idx = sp // num_graphs
        idy = sp  % num_graphs
        ax[idx,idy].clear()
        ax[idx,idy].plot(range(seq_len), np_inpt[:,sp], alpha=0.5, color='midnightblue')
        ax[idx,idy].plot(range(seq_len), neg_inpt[:,sp], alpha=0.5, color='orangered')
        ax[idx,idy].plot(range(seq_len), pos_inpt[:,sp], alpha=0.5, color='orangered')

def save(path, dict):
    if not os.path.exists(path):
        os.makedirs(path)

    for dk, data in dict.items():
        tdata = data
        torch.save(tdata, path + str(dk))

def make_toy_data(fct, n, dim_in, std = 1):
    inpt = torch.normal(mean=torch.zeros([n,dim_in]), std=std*torch.ones([n, dim_in]))
    label = fct(inpt)
    label = label - label.mean()
    label = 3*(label / (label.max()-label.min()))
    path = '/home/hendrik/Documents/master_project/LokalData/metaworld/test/toy_data/'
    save(path=path, dict={'inpt':inpt, 'label':label})
    return inpt, label