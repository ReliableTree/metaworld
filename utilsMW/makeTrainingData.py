from binascii import a2b_base64, a2b_hex
import imp
from sys import path
from time import sleep
from tkinter.messagebox import NO
import metaworld
import random
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
import torch
import numpy as np
import os
import numpy as np
import torch

from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor

class MakeTrainingData():
    def __init__(self, gt_policy, environment, training_examples) -> None:
        self.gt_policy = gt_policy
        self.training_examples = training_examples
        self.data = {}
        self.environment = environment

    def make_np_array(self, name, data):
        data = np.expand_dims(data, axis=0)
        if name not in self.data:
            self.data[name] = None
        if self.data[name] is None:
            self.data[name] = data
        else:
            self.data[name] = np.concatenate((self.data[name], data))

    def collect_training_data(self):
        gt_policy = self.gt_policy
        print(f'num training: {self.training_examples}')
        max_len = 0
        for scene in range(self.training_examples):
            env = self.environment()
            print(scene)
            self.data['obs_memory'] = None
            self.data['actions'] = None
            self.data['reward'] = None
            obs = env.reset()  # Reset environment

            for i in range(200):
                a = gt_policy.get_action(obs)
                obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
                obs_dict = gt_policy._parse_obs(obs)
                obs_arr = np.concatenate((obs_dict['hand_pos'], obs_dict['puck_pos'], obs_dict['puck_rot'], obs_dict['goal_pos']), axis=0)
                self.make_np_array(name='obs_memory', data=obs_arr)
                self.make_np_array(name='actions', data=a)
                self.make_np_array(name='reward', data = reward/10)
            print(f'reward:{reward}')
            if i > max_len:
                max_len = i
            if reward >= 9.5:
                self.make_np_array('training_data', data=self.data['obs_memory'])
                self.make_np_array('training_label', data=self.data['actions'])
                self.make_np_array('training_reward', data=self.data['reward'])
        print('max len {max_len}')
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for dk, data in self.data.items():
            tdata = torch.tensor(data)
            torch.save(tdata, path + str(dk))

class DefaultTraining():
    def __init__(self) -> None:
        pass
    
    def apply(self):
        gt_policy = SawyerPickPlaceV2Policy()
        door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["pick-place-v2-goal-observable"]

        mtd = MakeTrainingData(gt_policy, environment = door_open_goal_observable_cls, training_examples= 200)
        mtd.collect_training_data()
        mtd.save(path='/home/hendrik/Documents/master_project/LokalData/metaworld/pick-place/training_data/')

if __name__ == '__main__':
    DT = DefaultTraining()
    DT.apply()
    print('ASD')
