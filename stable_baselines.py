import gym
from stable_baselines3 import PPO
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from utilsMW.dataLoaderMW import TorchDatasetMWToy
import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance,  get_schedule_fn

from searchTest.toyEnvironment import check_outpt

import stable_baselines3

path = '/home/hendrik/Documents/master_project/LokalData/metaworld/small/train/'
train_data = TorchDatasetMWToy(path=path, device='cpu')

class my_env():
    def __init__(self, train_data):
        #obs = step, data, action, current_env
        self.observation_space = gym.spaces.box.Box(np.array([0, -2,-2,-2,-2, 0,0,0,0.,0]), np.array([6, 2,2,2,2, 1,1,1,1.,train_data.data.size(0)]), (10,), float)
        #next state (4)
        self.action_space = gym.spaces.box.Box(np.array([0,0,0,0]), np.array([1,1,1,1]), (4,), float)
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
        self.steps = 0
        self.current_env = -1
        self.data = train_data.data
        self.label = train_data.label
        self.traj = None
    def reset(self):
        self.traj = None
        self.current_env = (self.current_env + 1)%len(self.data)
        self.steps = 0
        last_action = torch.zeros(4, dtype=float)
        step = torch.tensor(self.steps)
        current_env = torch.tensor(self.current_env)
        data = self.data[self.current_env, 0]
        #label = self.label[self.current_env,0]
        state = torch.cat((step.view(1), data, last_action, current_env.view(1)), dim=0)
        return state

    def step(self, action):
        if type(action) is np.ndarray:
            action = torch.tensor(action)
        if self.traj is None:
            self.traj = action.reshape(1,-1)
        else:
            self.traj = torch.cat((self.traj, action.reshape(1,-1)), dim=0)



        self.steps += 1
        step = torch.tensor(self.steps)
        current_env = torch.tensor(self.current_env)

        #label = self.label[self.current_env, self.current_step]
        data = self.data[self.current_env, 0]

        state = torch.cat((step.view(1), data, action.reshape(-1), current_env.view(1)))
        if self.steps >= self.label.size(1):
            tol_neg = -0.55*torch.ones([self.traj.size(-1)])
            tol_pos = 0.7*torch.ones([self.traj.size(-1)])
            reward = int(check_outpt(self.label[self.current_env].unsqueeze(0), self.traj.unsqueeze(0), tol_neg=tol_neg, tol_pos=tol_pos))
            return (state, reward, True, {})
        else:
            return (state, 0., False, {})

    def close(self):
        pass
    
    def render(self, mode):
        pass

class toy_exper_model(OnPolicyAlgorithm):
    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]] = 'MlpPolicy',
            env: Union[GymEnv, str] = None,
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
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
            device: Union[th.device, str] = "auto",
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
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.data = train_data.data
        self.label = train_data.label


    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        step = int(obs.reshape(-1)[0])
        env = int(obs.reshape(-1)[-1])
        return self.label[env, step].reshape(1, -1), self.label[env, step].reshape(1, -1)
        
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


def sample_expert_transitions():
    expert = my_expert

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=10000),
    )
    return rollout.flatten_trajectories(rollouts)


if __name__ == '__main__':
    env = my_env(train_data=train_data)
    my_expert = toy_exper_model(train_data=train_data, env=env)
    from imitation.policies import *
    policy = ActorCriticPolicy(observation_space=env.observation_space, action_space=env.action_space, lr_schedule=lambda _: torch.finfo(torch.float32).max, net_arch = [dict(pi=[200, 200], vf=[200, 200])])
    transitions = sample_expert_transitions()
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=policy
    )
    
    for i in range(10):
        bc_trainer.train(n_epochs=50)
        rew = []
        for i in range(1000):
            obs = env.reset()
            done = False
            while not done:
                action, _ = bc_trainer.policy.predict(obs)
                obs, reward, done, _ = env.step(action=action)
            rew.append(reward)
        reward = torch.tensor(rew).type(torch.float).mean()
        print(reward)