import importlib
import sys

from tests.metaworld.envs.mujoco.sawyer_xyz import utils
sys.path.append('/home/hendrik/Documents/master_project/Code/LanguagePolicies/')
from model_src.modelTorch import PolicyTranslationModelTorch
from utils.networkTorch import NetworkTorch
import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable
import sys
import pickle
from time import sleep, time
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
import os
import numpy as np

from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor


class MWpolicy():
    def __init__(self, obs, device = 'cuda') -> None:
        model_path = "/home/hendrik/Documents/master_project/LokalData/Data/Model/nZ4zmE4oOmR/best_val/policy_translation_h"
        setup_path = "/home/hendrik/Documents/master_project/LokalData/Data/Model/nZ4zmE4oOmR/best_val/model_setup.pkl"
        with open(setup_path, 'rb') as f:
            model_setup = pickle.load(f)
        model_setup['use_memory'] = True
        model_setup['train']      = True
        print('load model')
        self.model = self.setupModel(device, model_path=model_path, model_setup=model_setup, obs=obs)

    def action(self, observation):
        a = self.model.forward(observation)
        return a

    def setupModel(self, device, model_path, model_setup, obs):
        model   = PolicyTranslationModelTorch(od_path="", model_setup=model_setup).to(device)
        model.forward(obs)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        return model

import time
class ApplyMWPolcy():
    def __init__(self) -> None:
        door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["pick-place-v2-goal-observable"]
        gt_policy = SawyerPickPlaceV2Policy()
        succ = np.zeros(0)
        env = door_open_goal_observable_cls()
        obs = env.reset()  # Step the environoment with the sampled random action
        obs_dict = gt_policy._parse_obs(obs)
        obs_arr = np.concatenate((obs_dict['hand_pos'], obs_dict['puck_pos'], obs_dict['puck_rot'], obs_dict['goal_pos']), axis=0)
        obs_arr = torch.tensor(obs_arr, dtype = torch.float).reshape(1,1,-1)
        mwPolicy = MWpolicy(obs_arr)
        for i in range(1000):
            env = door_open_goal_observable_cls()
            obs = env.reset()
            for j in range(200):
                a = gt_policy.get_action(obs)
                obs, reward, done, info = env.step(a)
            if reward >= 0.95:
                succ = np.append(succ, 1)
            else:
                succ = np.append(succ, 0)
        print(f'success rate final: {succ.mean()}')
        print(f'std final: {np.std(succ)}')
        succ = np.zeros(0)


        for i in range(1000):
            env = door_open_goal_observable_cls()
            obs = env.reset()  # Step the environoment with the sampled random action
            obs_dict = gt_policy._parse_obs(obs)
            obs_arr = np.concatenate((obs_dict['hand_pos'], obs_dict['puck_pos'], obs_dict['puck_rot'], obs_dict['goal_pos']), axis=0)
            obs_arr = torch.tensor(obs_arr, dtype = torch.float).reshape(1,1,-1)
            result = mwPolicy.model.forward(obs_arr)['gen_trj'].detach().numpy()
            for a in result[0]:
                obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
                #env.render()
            if reward >= 0.95:
                succ = np.append(succ, 1)
            else:
                succ = np.append(succ, 0)
        print(f'success rate final: {succ.mean()}')
        print(f'std final: {np.std(succ)}')


    def make_np_array(self, name, data):
        data = np.expand_dims(data, axis=0)
        if name not in self.data:
            self.data[name] = None
        if self.data[name] is None:
            self.data[name] = data
        else:
            self.data[name] = np.concatenate((self.data[name], data))
        
if __name__ == '__main__':
    AMP = ApplyMWPolcy()

