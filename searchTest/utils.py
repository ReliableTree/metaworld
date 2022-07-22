# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University
#matplotlib.use("TkAgg")
from select import select
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from hashids import Hashids
import os
import torch
try:
    from imitation.data import rollout
    from imitation.data.wrappers import RolloutInfoWrapper
except:
    pass
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from MetaWorld.searchTest.toyEnvironment import check_outpt
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
import sys
from pathlib import Path
parent_path = str(Path(__file__).parent.absolute())
parent_path += '/../../'
sys.path.append(parent_path)
from LanguagePolicies.utils.graphsTorch import TBoardGraphsTorch
from gym.wrappers import TimeLimit
global SAMPLED_ENVS
SAMPLED_ENVS = 0
global NUM_RESETS


def simulate(policy, n, val_env):
    rew = []
    for j in range(n):
        obs = val_env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, done, _ = val_env.step(action=action)

        rew.append(reward)
    reward = torch.tensor(rew).type(torch.float).mean()
    return reward

from prettytable import PrettyTable
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

def sample_expert_transitions(expert, env, n_episodes):
    print("Sampling expert transitions.")
    rollouts = rollout.generate_trajectories(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(n_episodes=n_episodes, n_timesteps=None),
    )
    return rollout.flatten_trajectories(rollouts)

def make_counter_embedding(x, bits):
    mask = 2**torch.arange(bits-1,-1,-1)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def get_integer(x):
    bits = len(x)
    mask = 2**torch.arange(bits-1,-1,-1)
    integer = (x*mask).sum()
    return int(integer)

def get_num_bits(interger):
    return int(torch.ceil(torch.log2(torch.tensor(interger))))

def make_obs_act_space(data, label):
    trj_len = label.size(1)
    bits = get_num_bits(trj_len)
    action_dim = label.size(-1)
    obs_dim = bits + (2*action_dim)
    obs_array_low = [0]*obs_dim
    obs_array_high = [1]*obs_dim
    action_low = [0]*action_dim
    action_high = [1]*action_dim
    print(f'obs: {obs_dim}')
    print(f'action_dim: {action_dim}')
    
    observation_space = gym.spaces.box.Box(np.array(obs_array_low), np.array(obs_array_high), (obs_dim,), float)
    #next state (4)
    action_space = gym.spaces.box.Box(np.array(action_low), np.array(action_high), (action_dim,), float)
    return observation_space, action_space


class MyEnv():
    @staticmethod
    def set_train_data(train_data, tol_neg, tol_pos, window, show_step):
        MyEnv.train_data = train_data
        MyEnv.tol_neg = tol_neg
        MyEnv.tol_pos = tol_pos
        MyEnv.window = window
        MyEnv.show_step = show_step

    def __init__(self, data=None):
        if data is None:
            data = MyEnv.train_data
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
        self.steps = 0
        self.current_env = -1
        self.data = data.data
        self.label = data.label
        self.observation_space, self.action_space = make_obs_act_space(self.data, self.label)
        self.traj = None
        self.num_envs = 1
        self.sampled_envs = 0

    def reset_envs_sampled(self):
        global SAMPLED_ENVS
        SAMPLED_ENVS = 0

    def reset(self, train=True):
        if train:
            global SAMPLED_ENVS
            SAMPLED_ENVS += 1
        self.sampled_envs = SAMPLED_ENVS
        self.traj = None
        self.current_env = (self.current_env + 1)%len(self.data)
        self.steps = 0
        last_action = torch.zeros(4, dtype=float, device=self.data.device)
        trj_len = self.label.size(1)
        bits = get_num_bits(trj_len)
        #print(f'bits: {bits}')
        step = make_counter_embedding(torch.tensor(self.steps, device=self.data.device), bits)
        if not MyEnv.show_step:
            step = torch.zeros_like(step)
        data = self.data[self.current_env, 0]
        state = torch.cat((step.reshape(-1), data, last_action), dim=0).type(torch.float).numpy()
        return state
        

    def step(self, action):
        if type(action) is np.ndarray:
            action = torch.tensor(action, device=self.data.device)
        if self.traj is None:
            self.traj = action.reshape(1,-1)
        else:
            self.traj = torch.cat((self.traj, action.reshape(1,-1)), dim=0)

        self.steps += 1
        trj_len = self.label.size(1)
        bits = get_num_bits(trj_len)
        step = make_counter_embedding(torch.tensor(self.steps, device=self.data.device), bits)
        #print(f'step: {step}')
        if not MyEnv.show_step:
            step = torch.zeros_like(step)
        data = self.data[self.current_env, 0]
        state = torch.cat((step.reshape(-1), data, action.reshape(-1)), dim=0).type(torch.float).numpy()
        if self.steps >= self.label.size(1):
            tol_neg = MyEnv.tol_neg*torch.ones([self.traj.size(-1)])
            tol_pos = MyEnv.tol_pos*torch.ones([self.traj.size(-1)])
            window = MyEnv.window
            reward = float(check_outpt(self.label[self.current_env].unsqueeze(0), self.traj.unsqueeze(0), tol_neg=tol_neg, tol_pos=tol_pos, window=window))
            return (state, reward, True, {})
        else:
            return (state, 0., False, {})

    def close(self):
        pass
    
    def render(self, mode):
        pass

class ToyExpertModel(OnPolicyAlgorithm):
    @staticmethod
    def set_datasets(train_data, val_data):
        ToyExpertModel.train_data = train_data
        ToyExpertModel.val_data = val_data
        
        env_id_train = dict()
        env_id_val = dict()
        for i, data in enumerate(train_data):
            env_id_train[str(data[0].reshape(-1).tolist())] = i
        for i, data in enumerate(val_data):
            env_id_val[str(data[0].reshape(-1).tolist())] = i
        ToyExpertModel.env_id_train = env_id_train
        ToyExpertModel.env_id_val = env_id_val

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]] = 'MlpPolicy',
            env: Union[GymEnv, str] = None,
            learning_rate = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range = 0.2,
            clip_range_vf = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
            _init_setup_model: bool = True,
            train_data = None
        ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
        )
        self.data = train_data.data
        self.label = train_data.label
        self.observation_space, self.action_space = make_obs_act_space(self.data, self.label)
        self.policy = self

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        trj_len = self.label.size(1)
        bits = get_num_bits(trj_len)

        obs = torch.tensor(obs).reshape(-1)
        step = get_integer(obs[:bits])

        env_key = str(obs.reshape(-1)[bits:bits+4].tolist())
        try:
            env = ToyExpertModel.env_id_train[env_key]
            result = ToyExpertModel.train_data.label[env, step].reshape(1, -1)
        except:
            env = ToyExpertModel.env_id_val[env_key]
            result = ToyExpertModel.val_data.label[env, step].reshape(1, -1).type(torch.float)

        return result, result

def draw_gt(val_env, tboard, stepid):
    target_trj = val_env.label[val_env.current_env]
    gen_trj = val_env.traj
    inpt = val_env.data[val_env.current_env][0]

    tol_neg = val_env.tol_neg*torch.ones([val_env.traj.size(-1)])
    tol_pos = val_env.tol_pos*torch.ones([val_env.traj.size(-1)])
    window = val_env.window

    tboard.plotDMPTrajectory(target_trj, gen_trj, torch.zeros_like(gen_trj),
                                None, None, None, stepid=stepid, save=False, name_plot='imitation baseline', path='',\
                                    tol_neg=tol_neg, tol_pos=tol_pos, inpt = inpt, name='imitation baseline', opt_gen_trj = None, window=window)


def benchmark_policy(policy, path, logname, eval_epochs, val_env, stepid, best_reward, save_model = True, do_draw_gt = False):
    tboard = TBoardGraphsTorch(logname=logname, data_path=path)
    rew = []
    success = []
    for j in range(eval_epochs):
        trj = []
        obs = val_env.reset()
        done = False
        k = 0
        while not done:
            k+=1
            action, _ = policy.predict(obs)
            obs, reward, done, info = val_env.step(action)
            if j == eval_epochs - 1:
                trj.append(action)
        rew.append(reward)
    reward = torch.tensor(rew).type(torch.float).mean()
    success = torch.tensor(success).type(torch.float).mean()
    trj = torch.tensor(np.array(trj))
    if len(trj.shape) == 1:
        trj = trj.reshape(-1,1)
    if reward >= best_reward and save_model:
        best_reward = reward
        save_path = path + logname
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(policy.state_dict(), save_path + '/best_model' + str(reward))
    #tboard.addValidationScalar('success rate', success.detach(), stepid=stepid)
    tboard.addValidationScalar('reward', reward.detach(), stepid=stepid)
    if do_draw_gt:
        draw_gt(val_env=val_env, tboard=tboard, stepid=stepid)
    else:
        trj = trj.transpose(0,1)
        print(trj.shape)
        tboard.plotDMPTrajectory(trj[0], trj[0], torch.zeros_like(trj[0]),
                            None, None, None, stepid=stepid, save=False, name_plot='logname')
    return best_reward

def train_policy(trainer, learn_fct, val_env, logname, path, n_epochs, n_steps, eval_epochs = 100, step_fct = None):
    if step_fct is None:
        global SAMPLED_ENVS
        step_id = SAMPLED_ENVS
    else:
        step_id = 0
    best_reward = -10
    for i in range(n_epochs):
        if step_fct is not None:
            step_id = step_fct(step_id)
        best_reward = benchmark_policy(policy = trainer.policy, path=path, logname=logname, eval_epochs=eval_epochs, val_env=val_env, stepid=step_id, best_reward=best_reward)
        learn_fct(n_epochs=n_steps)

class LearnWrapper():
    def __init__(self, trainer):
        self.trainer = trainer
    
    def train(self, n_epochs):
        self.trainer.learn(total_timesteps=n_epochs, log_interval=140000)

NUM_RESETS = 0
class SuperMyGymWrapper():
    def __init__(self, tag, bo=None) -> None:
        self.tag = tag
        self.bo=bo

    def make_wrapper(self, count_resets = True):

        if self.bo is None:
            bo = gym.make(self.tag)
        else:
            bo = self.bo

        class MyGymWrapper(bo.__class__):
            def __init__(self, baseObject, count_resets):
                self.__class__ = type(baseObject.__class__.__name__,
                                    (self.__class__, baseObject.__class__),
                                    {})
                self.__dict__ = baseObject.__dict__
                self.count_resets = count_resets

            def reset_count(self):
                global NUM_RESETS
                NUM_RESETS = 0

            def reset(self):
                global NUM_RESETS
                if self.count_resets:
                    NUM_RESETS += 1
                    print(f'num_ sampled: {NUM_RESETS}')
                return super().reset()

            def step(self, action):
                obsv, rew, done, info = super().step(action)
                return obsv, rew, done, info

            
        mgw = MyGymWrapper(baseObject=bo, count_resets=count_resets)
        return mgw     

class supermipWrapper():
    def __init__(self, tag, bo=None) -> None:
        self.tag = tag
        self.bo=bo

    def make_wrapper(self, count_resets = True):

        if self.bo is None:
            bo = gym.make(self.tag)
        else:
            bo = self.bo

        class MyGymWrapper(bo.__class__):
            def __init__(self, baseObject, count_resets):
                self.__class__ = type(baseObject.__class__.__name__,
                                    (self.__class__, baseObject.__class__),
                                    {})
                self.__dict__ = baseObject.__dict__
                self.count_resets = count_resets

            def reset_count(self):
                global NUM_RESETS
                NUM_RESETS = 0

            def reset(self):
                global NUM_RESETS
                if self.count_resets:
                    NUM_RESETS += 1
                    print(f'num_ sampled: {NUM_RESETS}')
                return super().reset()

            def _rebuild_obsv(self, obsv, info):
                obsvs = []
                for i in range(len(info)):
                    obsv_dict = {}
                    for key in obsv:
                        obsv_dict[key] = obsv[key][i]
                    obsvs.append(obsv_dict)
                return obsvs


            def step(self, action):
                obsv, rew, done, info = super().step(action)
                obsv = self._rebuild_obsv(obsv=obsv, info=info)
                return obsv, rew, done, info

        mgw = MyGymWrapper(baseObject=bo, count_resets=count_resets)
        return mgw       

class my_dummy_wrapper:
    def __init__(self, tag) -> None:
        self.tag = tag

    def get_env(self):
        return gym.make(self.tag)