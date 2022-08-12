from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import numpy as np
import torch


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
    def __init__(self) -> None:
        pass

    


    def get_env(self, n, env_tag):
        envs = []
        env_name = self.policy_dict[env_tag][1]
        gt_policy = self.policy_dict[env_tag][0]
        for i in range(n):
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
            obs_dict = gt_policy._parse_obs(obs)
            obs_arr = np.concatenate((obs_dict['hand_pos'], obs_dict['puck_pos'], obs_dict['puck_rot'], obs_dict['goal_pos']), axis=0)
            obs_arr = torch.tensor(obs_arr, dtype = torch.float, device = policy.device).reshape(1,1,-1)
            result_pol = policy.forward(obs_arr)
            result = result_pol['gen_trj'].detach()
            f_result = result_pol['inpt_trj'].detach()
            np_result = result.cpu().detach().numpy()

            for a in np_result[0]:
                obs, reward_policy, done, info = env.step(a) 
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