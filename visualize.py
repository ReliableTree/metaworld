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
from imitation.algorithms.adversarial.airl import AIRL

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
import os

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
import time

pape.reset()

for i in range(10000):
    pape.render()
    time.sleep(0.01)