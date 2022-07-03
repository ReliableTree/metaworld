import imp
from sys import path
from time import sleep

from cv2 import VideoWriter
import metaworld
import random
#from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
#from metaworld.policies.sawyer_basketball_v2_policy import SawyerBasketballV2Policy
#from metaworld.policies.sawyer_assembly_v2_policy import SawyerAssemblyV2Policy
#from metaworld.policies.sawyer_box_close_v2_policy import SawyerBoxCloseV2Policy
from metaworld.policies import *
import torch
import numpy as np
import os
import numpy as np
from torch.utils.data import DataLoader
from gym.wrappers import RecordVideo

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

    def collect_training_data(self, render = False, steps=500):
        import time
        gt_policy = self.gt_policy
        #print(f'num training: {self.training_examples}')
        reward = 0
        tries = 0
        while reward < 10 and tries < 10:
            tries +=1 
            for scene in range(self.training_examples):
                env = self.environment()
                #print(scene)
                self.data['obs_memory'] = None
                self.data['actions'] = None
                self.data['reward'] = None
                obs = env.reset()  # Reset environment
                step = 0
                reward = 0
                while step < steps and reward < 10:
                    step += 1
                    a = gt_policy.get_action(obs)
                    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
                    if render:
                        time.sleep(0.05)
                        env.render()
                    obs_dict = gt_policy._parse_obs(obs)
                    obs_arr = np.zeros(14)
                    pointer = 0
                    for key in obs_dict:
                        if not (('unused' in key) or ('extra' in key) or ('_prev_obs' in key)):
                            len_data = obs_dict[key].shape
                            if len(len_data) != 0:
                                len_data = len_data[0]
                            else:
                                len_data = 1
                            obs_arr[pointer:pointer+len_data] = obs_dict[key]
                            pointer += len_data

                    #self.make_np_array(name='obs_memory', data=obs_arr)
                    #self.make_np_array(name='actions', data=a)
                    #self.make_np_array(name='reward', data = reward/10)
                print(f'reward:{reward}')
            #if i > max_len:
            #    max_len = i
            #if reward >= 9.5:
            #    self.make_np_array('training_data', data=self.data['obs_memory'])
            #    self.make_np_array('training_label', data=self.data['actions'])
            #    self.make_np_array('training_reward', data=self.data['reward'])
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for dk, data in self.data.items():
            tdata = torch.tensor(data)
            torch.save(tdata, path + str(dk))

class DefaultTraining():
    def __init__(self) -> None:
        pass
    
    def apply(self, scene, policy):

        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[scene]

        mtd = MakeTrainingData(policy, environment = env, training_examples= 20000)
        mtd.collect_training_data(render=True)
        #path = '/home/hendrik/Documents/master_project/LokalData/metaworld/pick-place/training_data/'
        path = '/home/hendrik/Documents/master_project/LokalData/metaworld/test/training_data/'
        mtd.save(path=path)

def make_policy_dict():
    policy_dict = {}
    for key in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        string_arr = key.split('-')
        v2_ind = string_arr.index('v2')
        string_arr = string_arr[:v2_ind]
        policy_name = ''
        for i, string in enumerate(string_arr):
            policy_name+=string
            string_arr[i] = string.capitalize()
        #policy_name = 'policy_dict["' + key_replaced + '_policy"] = [Sawyer'
        entry = 'policy_dict["' + str(policy_name) + '"] = [Sawyer'
        for string in string_arr:
            entry += string
        entry += 'V2Policy(), "' + key + '"]'
        try:
            exec(entry)
        except (NameError):
            pass
    return policy_dict


class SuccessSimulation():
    def __init__(self, device='cpu') -> None:
        self.policy_dict = make_policy_dict()
        self.window=0


    def get_env(self, n, env_tag, name):
        envs = []
        env_name = self.policy_dict[env_tag][1]
        gt_policy = self.policy_dict[env_tag][0]
        for i in range(n):
            def st(i):
                return True
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
            envs.append([env, gt_policy])
        return envs

    def get_success(self, policy, envs):
        trajectories = []
        inpt_obs = []
        success = []
        labels = []
        f_results = []
        for env_tuple in envs:
            env, gt_policy = env_tuple
            obs = env.reset()  # Reset environment
            def asd(i):
                return True
            obs_dict = gt_policy._parse_obs(obs)
            obs_arr = np.concatenate((obs_dict['hand_pos'], obs_dict['puck_pos'], obs_dict['puck_rot'], obs_dict['goal_pos']), axis=0)
            obs_arr = torch.tensor(obs_arr, dtype = torch.float, device = policy.device).reshape(1,1,-1)
            result_pol = policy.forward(obs_arr)
            result = result_pol['gen_trj'].detach()
            f_result = result_pol['inpt_trj'].detach()
            np_result = result.cpu().detach().numpy()

            for a in np_result[0]:
                obs, reward_policy, done, info = env.step(a)  # Step the environoment with the sampled random action
                #env.render()
            obs = env.reset()  # Reset environment
            steps = 0
            label = []
            while steps < 200:
                steps += 1
                a = gt_policy.get_action(obs)
                obs, reward, done, info = env.step(a)
                label.append(torch.tensor(a, dtype=torch.float32))
            if reward > 0.95:
                trajectories += [result]
                inpt_obs.append(obs_arr)
                if reward_policy >= 0.95:
                    success.append(torch.ones(1, device=policy.device, dtype=torch.bool))
                else:
                    success.append(torch.zeros(1, device=policy.device, dtype=torch.bool))
                label = torch.cat([*label], dim = 0).reshape(-1,a.shape[0])
                labels.append(label)
                f_results += [f_result]
        if len(label) > 0:
            labels = torch.cat([*labels], dim=0).reshape(-1, labels[0].size(0), labels[0].size(1)).to(policy.device)
            trajectories = torch.cat([*trajectories], dim=0)
            inpt_obs = torch.cat([*inpt_obs], dim=0)
            success = torch.cat([*success], dim=0)
            f_results = torch.cat([*f_results], dim=0)

            return trajectories, inpt_obs, labels, success, f_results
        else:
            return False


class ToySimulation():
    def __init__(self, neg_tol, pos_tol, check_outpt_fct, dataset, window = 9) -> None:
        self.dataset = dataset     
        self.neg_tol = neg_tol
        self.pos_tol = pos_tol
        self.check_outpt_fct = check_outpt_fct
        self.window = window

    def get_env(self, n, env_tag):
        indices = torch.randperm(len(self.dataset))[:n]
        return indices

    def get_success(self, policy, envs):
        trajectories = []
        f_results = []
        inpt_obs = []
        success = []
        labels = []
        subset = torch.utils.data.Subset(self.dataset, envs)
        dataloader = DataLoader(subset, batch_size=200, shuffle=False)
        for inpt, label in dataloader:
            result_pol = policy.forward(inpt)
            result = result_pol['gen_trj'].detach()
            f_result = result_pol['inpt_trj'].detach()
            trajectories += [result]
            f_results += [f_result]
            inpt_obs.append(inpt[:,:1])
            success.append(self.check_outpt_fct(label=label, outpt=result, tol_neg=self.neg_tol, tol_pos=self.pos_tol, window=self.window))
            labels.append(label)

        trajectories = torch.cat([*trajectories], dim=0)
        f_results = torch.cat([*f_results], dim=0)
        inpt_obs = torch.cat([*inpt_obs], dim=0)
        success = torch.cat([*success], dim=0)
        labels = torch.cat([*labels], dim=0)

        return trajectories, inpt_obs, labels, success, f_results

if __name__ == '__main__':
    DT = DefaultTraining()
    policy_dict = make_policy_dict()
    print(policy_dict)
    '''max_len_a = 0
    for policy in policy_dict:
        gt_policy = policy_dict[policy][0]
        env_name = policy_dict[policy][1]
        print(env_name)
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
        MTD = MakeTrainingData(gt_policy, env, 1)
        la = MTD.collect_training_data(steps= 500)'''


