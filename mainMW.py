
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
from utilsMW.dataLoaderMW import TorchDatasetMW
from torch.utils.data import DataLoader
from gym import logger



# Learning rate for the adam optimizer
LEARNING_RATE   = 0.001
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

def setupModel(device , epochs ,  batch_size, path_dict , logname , model_path, tboard, model_setup, train_size = 1):
    model   = PolicyTranslationModelTorch(od_path="", model_setup=model_setup, device=device).to(device)

    tailor_model = TailorTransformer(model_setup=model_setup['tailor_transformer'])
    tailor_models = [tailor_model]
    #tailor_models=[]
    train_data = TorchDatasetMW(path=path_dict['META_WORLD'], device=device)

    print(f'len(train_data): {len(train_data)}')

    #train_data = torch.utils.data.Subset(train_data, train_indices).to(device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    #eval_data = TorchDataset(path = path_dict['VAL_DATA_TORCH'], device=device).to(device)
    eval_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    env_tag = 'pickplace'
    network = NetworkMeta(model, tailor_models=tailor_models, env_tag=env_tag, data_path=path_dict['DATA_PATH'],logname=logname, lr=LEARNING_RATE, lw_atn=WEIGHT_ATTN, lw_w=WEIGHT_W, lw_trj=WEIGHT_TRJ, lw_gen_trj = WEIGHT_GEN_TRJ, lw_dt=WEIGHT_DT, lw_phs=WEIGHT_PHS, lw_fod=0, gamma_sl = 1, device=device, tboard=tboard)
    network.setDatasets(train_loader=train_loader, val_loader=eval_loader)

    network.setup_model(model_params=model_setup)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    count_parameters(network)

    #print(f'number of param,eters in net: {len(list(network.parameters()))} and number of applied: {i}')
    #network.load_state_dict(torch.load(MODEL_PATH), strict=True)
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
        'TRAIN_DATA_TORCH' : os.path.join(data_path, 'TorchDataset/train_data_torch.txt'),
        'VAL_DATA_TORCH' : os.path.join(data_path, 'TorchDataset/val_data_torch.txt'),
        'TRAIN_DATA' : os.path.join(data_path, 'GDrive/train.tfrecord'),
        'VAL_DATA' : os.path.join(data_path, 'GDrive/validate.tfrecord'),
        'GLOVE_PATH' : os.path.join(data_path, 'GDrive/glove.6B.50d.txt'),
        'META_WORLD' : os.path.join(data_path, 'metaworld/pick-place/training_data/'),
        'DATA_PATH' : data_path
        }

        device = 'cuda'
        if '-device' in args:
            device = args[args.index('-device') + 1]
        from utilsMW.model_setup import model_setup
        model_path = None
        if '-model' in args:
            model_path = args[args.index('-model') + 1] + 'policy_translation_h'
            if '-model_setup' in args:
                setup_path = args[args.index('-model') + 1] + 'model_setup.pkl'
                with open(setup_path, 'rb') as f:
                    model_setup = pickle.load(f)
                model_setup['train']      = True
                print('load model')

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
        network = setupModel(device=device, epochs = epochs, batch_size = batch_size, path_dict = path_dict, logname=logname, model_path=model_path, tboard=tboard, model_setup=model_setup, train_size=train_size)
        print(f'end saving: {path_dict["MODEL_PATH"]}')
        torch.save(network.state_dict(), path_dict['MODEL_PATH'])

