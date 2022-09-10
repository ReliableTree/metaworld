from random import seed
from typing import Optional, Type, Union

import gym
import numpy as np
import tensorflow as tf
import torch
import torch as th
from gym.wrappers import TimeLimit
from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from RlBaselines3Zoo import enjoy
from sb3_contrib.tqc.policies import MultiInputPolicy
from sb3_contrib.tqc.tqc import TQC
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (ActorCriticPolicy, BaseModel,
                                               MultiInputActorCriticPolicy)
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   CombinedExtractor,
                                                   FlattenExtractor, NatureCNN,
                                                   create_mlp,
                                                   get_actor_critic_arch)
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ppo import MlpPolicy

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from MetaWorld.metaworld.policies.sawyer_pick_place_v2_policy import \
    SawyerPickPlaceV2Policy
from MetaWorld.searchTest.utils import (LearnWrapper, MyEnv, ToyExpertModel,
                                        VecExtractor, benchmark_policy,
                                        parse_sampled_transitions,
                                        sample_expert_transitions,
                                        train_policy)
from MetaWorld.utilsMW.makeTrainingData import make_policy_dict
from MetaWorld.utilsMW.model_setup_obj import NetworkSetup
from MetaWorld.utilsMW.trainer import ActiveCritic, ActiveCriticArgs
from searchTest.utils import (MyEnv, ToyExpertModel, benchmark_policy,
                              get_num_bits, make_counter_embedding,
                              sample_expert_transitions, train_policy)
from utilsMW.dataLoaderMW import TorchDatasetMWToy

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

policy_dict = make_policy_dict()
env_tag = 'pickplace'
gt_policy = policy_dict[env_tag]
pape = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_tag][1]]()
pape._freeze_rand_vec = False
timelimit = TimeLimit(env=pape, max_episode_steps=100)
dv1 = DummyVecEnv([lambda: RolloutInfoWrapper(timelimit)])


class ImitationLearningWrapper:
    def __init__(self, policy, env: GymEnv):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.policy = policy

    def predict(self, obsv, deterministic=None):
        actions = []
        for obs in obsv:
            actions.append(self.policy.get_action(obs))
        return actions


IGTP = ImitationLearningWrapper(policy=gt_policy[0], env=dv1).predict


def new_epoch(current_obs, check_obsvs):
    result = not th.equal(current_obs.reshape(-1)
                          [36:39], check_obsvs.reshape(-1)[36:39])
    return result


class DummyExtractor:
    def __init__(self):
        pass

    def forward(self, features):
        if type(features) is np.ndarray:
            features = th.tensor(features)
        return features


def sample_expert_transitions(expert):
    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        dv1,
        rollout.make_sample_until(min_timesteps=None, min_episodes=10),
        unwrap=True,
        exclude_infos=False,
    )
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions(IGTP)



def new_epoch(current_obs, check_obsvs):
    result = not th.equal(current_obs.reshape(-1)
                          [36:39], check_obsvs.reshape(-1)[36:39])
    return result


class DummyExtractor:
    def __init__(self):
        pass

    def forward(self, features):
        if type(features) is np.ndarray:
            features = th.tensor(features)
        return features

from itertools import count
from select import epoll


class NoObsvSparseReward(gym.Env):
    def __init__(self, env:gym.Env, seq_len, device = 'cuda', seed=None) -> None:
        super().__init__()
        if seed is None:
            env = env()
            env._freeze_rand_vec = False
            self.env = TimeLimit(env=env, max_episode_steps=seq_len)
            self.env._freeze_rand_vec = False

        self.action_space = self.env.action_space
        self.bits = get_num_bits(self.env._max_episode_steps+1)
        lows = self.env.observation_space.low
        lows = np.append(lows, np.zeros(self.bits))
        highs = self.env.observation_space.high
        highs = np.append(highs, np.ones(self.bits))
        self.observation_space = gym.spaces.Box(low=lows, high=highs, dtype=self.env.observation_space.dtype)
        self.device = device

    def reset(self):
        response = self.env.reset()
        self.response = response
        step = make_counter_embedding(torch.tensor(self.env._elapsed_steps), self.bits)
        response = np.append(response, step)

        return response

    def step(self, action):
        obs, rew, done, info = self.env.step(action=action)
        info['obs'] = obs
        step = make_counter_embedding(torch.tensor(self.env._elapsed_steps), self.bits)
        obs = np.append(self.response, step)
        if done:
            rew = info['success']
        else:
            rew = 0

        return obs, rew, done, info

def strip_transitions(transitions:types.Transitions):
    observations = np.array(transitions.obs)
    striped_observations = strip_obs(observations)
    next_observations = np.array(transitions.obs)
    next_striped_observations = strip_obs(next_observations, offset=1)
    new_dict = {
        'obs':striped_observations,
        'next_obs':next_striped_observations,
        'dones':transitions.dones,
        'infos':transitions.infos,
        'acts':transitions.acts
    }
    return types.Transitions(**new_dict)


def strip_obs(obs, offset = 0):
    dones = np.where(transitions.dones)[0]
    if len(dones) > 1:
        epch_len = dones[1] - dones[0]
    else:
        epch_len = len(obs)
    start_points = dones - epch_len + 1
    proto_obs = obs[start_points]
    proto_obs = np.expand_dims(proto_obs, axis=0)

    num_bits = get_num_bits(epch_len+1)
    counter_embedding = make_counter_embedding(torch.arange(offset, epch_len+offset), num_bits)
    counter_embedding = np.expand_dims(counter_embedding, axis=0)
    counter_embedding = np.repeat(counter_embedding, np.sum(transitions.dones, dtype=np.int32), axis=0)
    counter_embedding = counter_embedding.reshape(-1, counter_embedding.shape[-1])
    proto_obs = np.repeat(proto_obs, epch_len, axis=0)
    proto_obs = proto_obs.transpose([1,0,2])
    proto_obs = proto_obs.reshape(-1,proto_obs.shape[-1])
    proto_obs = np.concatenate((proto_obs, counter_embedding), axis=1)
    return proto_obs

stripped_transitions = strip_transitions(transitions)


learner = SAC(
    env=dv1,
    policy='MlpPolicy',
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    verbose=-1,
)


reward_net = BasicRewardNet(
    dv1.observation_space, dv1.action_space, normalize_input_layer=RunningNorm, 
)

gail_trainer = GAIL(
    demonstrations=transitions,
    demo_batch_size=100,
    gen_replay_buffer_capacity=100,
    n_disc_updates_per_round=4,
    venv=dv1,
    gen_algo=learner,
    reward_net=reward_net,
)

gail_trainer.train(30000)