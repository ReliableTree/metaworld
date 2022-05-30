import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
def make_toy_data(num_examples, inpt_func, outpt_func):
    generator =  torch.rand(num_examples)
    inpt = inpt_func(generator)
    outpt = outpt_func(inpt)
    return inpt, outpt

def check_outpt(label, outpt, tol_neg, tol_pos, window = 0):
    if window > 0:
        tol_neg, tol_pos, inpt= make_sliding_tol(label=inpt, neg_tol=tol_neg, pos_tol=tol_pos, window=window)

    diff = outpt - label
    

    if window > 0:
        neg_acc = diff > tol_neg
        pos_acc = diff < tol_pos
    else:
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
    tri = torch.randint(low=0, high=dim_in, size=(dim_out, dim_in))
    signs = torch.randint(low=-1, high=2, size=(dim_in*dim_out,)).reshape(dim_out, dim_in)

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

def plot_fcts(inpt, neg_tol, pos_tol, window = 0):
    inpt_dim = len(inpt[0])
    if window==0:
        neg_inpt = (inpt + neg_tol[None,:]).numpy()
        pos_inpt = (inpt + pos_tol[None,:]).numpy()
    else:
        sliding_tol_pos, sliding_tol_neg, inpt= make_sliding_tol(label=inpt.unsqueeze(0), neg_tol=neg_tol, pos_tol=pos_tol, window=window)
        pos_inpt, neg_inpt = sliding_tol_pos.numpy(), sliding_tol_neg.numpy()
        inpt = inpt[0]

    np_inpt = inpt.numpy()
    print(f'input: {np_inpt.shape}')
    
    seq_len = len(inpt)
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

def make_toy_data(fct, n, dim_in, std = 0.1, data_path= None):
    inpt = torch.normal(mean=torch.zeros([n,1, dim_in]), std=std*torch.ones([n,1,dim_in]))
    print(inpt.max())
    print(inpt.min())
    label = fct(inpt[:,0])
    tl = label.transpose(0,2)
    tl = tl.reshape([tl.size(0),-1])
    mean_label = tl.mean(dim=1)
    label = label - mean_label
    tl = label.transpose(0,2)
    tl = tl.reshape([tl.size(0),-1])
    tmax = tl[:10].max(dim=1)[0]
    tmin = tl[:10].min(dim=1)[0]
    rel_corr = tmax - tmin
    label = label % 1

    data_sets = ['train', 'val', 'test']
    if data_path is not None:
        for i, set in enumerate(data_sets):
            path =  data_path + set + '/'
            start = i * int(len(inpt)/len(data_sets))
            end = (i+1) * int(len(inpt)/len(data_sets))
            print(start)
            print(end)
            save(path=path, dict={'inpt':5*inpt[start:end], 'label':label[start:end]})
    return inpt, label

def make_sliding_tol_dim(label, window = 9, pos = True):
    batch_size = label.size(0)
    batch_counter = torch.arange(batch_size)
    counter = torch.arange(label.size(-1) - window) + int(window/2)
    window_counter = torch.arange(window) - int(window/2)
    s_ind = counter.repeat([batch_size,window,1]).transpose(-1,-2)
    f_ind = (counter[:,None] + window_counter[None,:]).repeat([batch_size, 1,1])
    batch_ind = batch_counter.reshape(-1,1,1).repeat([1,f_ind.size(-2), f_ind.size(-1)])
    ind = tuple((batch_ind, f_ind, s_ind))
    label_repeated = label.unsqueeze(-1).repeat([1,1,label.size(-1)])
    label_ind = label_repeated[ind]
    result = label_ind.max(dim=-1)[0], label_ind.min(dim=-1)[0]
    return result

def make_sliding_tol(label, neg_tol, pos_tol, window=9):
    tols_pos, tols_neg = [], []
    for dim in range(label.size(-1)):
        tol_pos, tol_neg = make_sliding_tol_dim(label=label[:,:,dim], window=window)
        tols_pos.append(tol_pos.unsqueeze(-1))
        tols_neg.append(tol_neg.unsqueeze(-1))
    sliding_tol_pos, sliding_tol_neg = torch.cat(tuple(tols_pos), dim=-1), torch.cat(tuple(tols_neg), dim=-1)
    neg_inpt = (sliding_tol_neg[0] + neg_tol[None,:])
    pos_inpt = (sliding_tol_pos[0] + pos_tol[None,:])
    inpt = label[:, int(window/2):-(int(window/2) + 1)]
    result = pos_inpt, neg_inpt, inpt
    return result

if __name__ == '__main__':
    args = sys.argv[1:]
    if '-path' not in args:
        print('no path given, not executing code')
    else:    
        data_path = args[args.index('-path') + 1]
        device = 'cuda'
        if '-device' in args:
            device = args[args.index('-device') + 1]
        
        dim_in = 4
        dim_out = 4
        seq_len = 200
        n = 30000
        if '-dim_in' in args:
            dim_in = int(args[args.index('-dim_in') + 1])
        if '-dim_out' in args:
            dim_out = int(args[args.index('-dim_out') + 1])
        if '-seq_len' in args:
            seq_len = int(args[args.index('-seq_len') + 1])
        if '-n' in args:
            n = int(args[args.index('-n') + 1])

        inner_fct = make_func(dim_in=dim_in, dim_out=dim_out, seq_len=seq_len)
        inpt, label = make_toy_data(inner_fct, n=n, dim_in=dim_in, std=0.1, data_path=data_path)
