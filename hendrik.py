
from __future__ import absolute_import, division, print_function, unicode_literals
import importlib
import sys

from tests.metaworld.envs.mujoco.sawyer_xyz import utils
sys.path.append('/home/hendrik/Documents/master_project/Code/LanguagePolicies/')
from model_src.modelTorch import PolicyTranslationModelTorch
from utils.Transformer import TailorTransformer
from utils.networkMeta import NetworkMeta
from utils.networkTorch import NetworkTorch
import hashids
import time
import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable
import sys
import pickle
from utilsMW.model_setup import model_setup
from utilsMW.dataLoaderMW import TorchDatasetMWToy
from torch.utils.data import DataLoader
from gym import logger
from utilsMW.makeTrainingData import ToySimulation
from searchTest.toyEnvironment import make_func, check_outpt, make_tol
from os import path, makedirs


# Learning rate for the adam optimizer
LEARNING_RATE   = 5e-5
META_LEARNING_RATE = 1e-4
LR_META_OPTIMIZED = 1
# Weight for the attention loss
WEIGHT_ATTN     = 1.0
# Weight for the motion primitive weight loss
WEIGHT_W        = 1.0
# Weight for the trajectroy generation loss
WEIGHT_TRJ      = 1#5.0

WEIGHT_GEN_TRJ  = 50

# Weight for the time progression loss
WEIGHT_DT       = 14.0
# Weight for the phase prediction loss
WEIGHT_PHS      = 1 #1.0


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def setupModel(device , epochs ,  batch_size, path_dict , logname , model_path, tboard, model_setup, train_size, load_tol = False):
    train_path = path_dict['META_WORLD'] + 'train/'
    val_path = path_dict['META_WORLD'] + 'val/'
    test_path = path_dict['META_WORLD'] + 'test/'
    tol_path = path_dict['META_WORLD'] + 'tol/'
    train_data = TorchDatasetMWToy(path=train_path, device=device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = TorchDatasetMWToy(path=val_path, device=device)
    eval_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_data = TorchDatasetMWToy(path=test_path, device=device)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    env_tag = 'pickplace'
    for inpt, outpt in train_loader:
        dim_in = inpt.size(1)
        dim_out = outpt.size(-1)
        seq_len = outpt.size(1)
        break
    model_setup['plan_nn']['plan']['d_output'] = dim_out
    model_setup['plan_nn']['plan']['seq_len'] = seq_len
    model   = PolicyTranslationModelTorch(od_path="", model_setup=model_setup, device=device).to(device)
    for inpt, outpt in train_loader:
        result = model(inpt)
        break

    if load_tol:
        with open(tol_path + 'tol.pkl', 'rb') as f:
            tol_neg, tol_pos = pickle.load(f)
    else:
        if not path.exists(tol_path):
            makedirs(tol_path)
        #tol_neg, tol_pos = make_tol(std_dev=5e-5, dim=dim_out, add=3e-1, device='cuda')
        tol_neg = -0.15*torch.ones([dim_out], device='cuda')
        tol_pos = 0.25*torch.ones([dim_out], device='cuda')
        with open(tol_path + 'tol.pkl', 'wb') as f:
            pickle.dump((tol_neg, tol_pos), f)  

    successSimulation = ToySimulation(neg_tol=tol_neg, pos_tol=tol_pos, check_outpt_fct=check_outpt, dataset=test_data, window = 19)

    model_setup['tailor_transformer']['seq_len'] = seq_len
    tailor_model = TailorTransformer(model_setup=model_setup['tailor_transformer'])
    
    
    network = NetworkMeta(model, tailor_models=[tailor_model], env_tag=env_tag, successSimulation=successSimulation, data_path=path_dict['DATA_PATH'],logname=logname, lr=LEARNING_RATE, mlr=META_LEARNING_RATE, mo_lr=LR_META_OPTIMIZED,  lw_atn=WEIGHT_ATTN, lw_w=WEIGHT_W, lw_trj=WEIGHT_TRJ, lw_gen_trj = WEIGHT_GEN_TRJ, lw_dt=WEIGHT_DT, lw_phs=WEIGHT_PHS, lw_fod=0, gamma_sl = 0.98, device=device, tboard=tboard)
    network.setDatasets(train_loader=train_loader, val_loader=eval_loader)

    network.setup_model(model_params=model_setup)
    if model_path is not None:
        network.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        #model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    count_parameters(network)
    #print('in tailor transfo:')
    #count_parameters(tailor_model)

    network.train(epochs=epochs, model_params=model_setup)
    return network
import os
if __name__ == '__main__':
    logger.set_level(40)
    args = sys.argv[1:]
    if '-path' not in args:
        print('no path given, not executing code')
    else:    
        data_path = args[args.index('-path') + 1]
        path_dict = {
        'META_WORLD' : os.path.join(data_path, 'metaworld/serverTest/'),
        'DATA_PATH' : data_path
        }
        print(path_dict)
        device = 'cuda'
        if '-device' in args:
            device = args[args.index('-device') + 1]
        from utilsMW.toy_model_setup import model_setup
        model_path = None
        if '-model' in args:
            model_path = args[args.index('-model') + 1] + 'policy_network'
            if '-model_setup' in args:
                setup_path = args[args.index('-model') + 1] + 'model_setup.pkl'
                with open(setup_path, 'rb') as f:
                    model_setup = pickle.load(f)
                model_setup['train']      = True
                print('load model')
        model_setup = model_setup
        
        epochs = 200
        if '-epochs' in args:
            epochs = int(args[args.index('-epochs') + 1])

        batch_size = 16
        if '-batch_size' in args:
            batch_size = int(args[args.index('-batch_size') + 1])

        tboard = True
        if '-tboard' in args:
            tboard = (args[args.index('-tboard') + 1]) == 'True'
            print(f'tboard: {tboard}')

        train_size = 1
        if '-train_size' in args:
            train_size = float(args[args.index('-train_size') + 1])

        hid             = hashids.Hashids()
        logname         = hid.encode(int(time.time() * 1000000))
        print(f'logname: {logname}')
        network = setupModel(device=device, epochs = epochs, batch_size = batch_size, path_dict = path_dict, logname=logname, model_path=model_path, tboard=tboard, model_setup=model_setup, train_size=train_size, load_tol=False)


