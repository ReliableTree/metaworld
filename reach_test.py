from MetaWorld.utilsMW.trainer import ActiveCritic, ActiveCriticArgs
from stable_baselines3.common.torch_layers import CombinedExtractor
from MetaWorld.utilsMW.model_setup_obj import NetworkSetup
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from MetaWorld.searchTest.utils import MyEnv, ToyExpertModel, sample_expert_transitions, benchmark_policy, LearnWrapper, train_policy, LearnWrapper, parse_sampled_transitions, VecExtractor
import torch
import zipfile
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import importlib
import numpy as np
from RlBaselines3Zoo import enjoy
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from stable_baselines3.common.policies import MultiInputActorCriticPolicy, ActorCriticPolicy, BaseModel
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from sb3_contrib.tqc.tqc import TQC
from sb3_contrib.tqc.policies import MultiInputPolicy
import numpy as np
import torch as th
import os

from MetaWorld.searchTest.utils import train_policy

import tensorflow as tf

torch.manual_seed(0)
np.random.seed(0)

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

model = enjoy.main(inpt_args="--algo tqc --env FetchPickAndPlace-v1 --folder /home/hendrik/Documents/master_project/Code/RlBaselines3Zoo/rl-trained-agents -n 300 --ret_model True")
def sample_expert_transitions():
    expert = model

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(model.env.envs[0])]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=100),
        unwrap=True,
        exclude_infos=False,
    )
    return rollout.flatten_trajectories(rollouts)

tp = '/home/hendrik/Documents/master_project/LokalData/ImitationLearning/transitions_rew_100'

#transitions = sample_expert_transitions()
#torch.save(transitions, tp)
transitions = torch.load('/home/hendrik/Documents/master_project/LokalData/ImitationLearning/transitions_rew_100')

from MetaWorld.metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from MetaWorld.utilsMW.makeTrainingData import make_policy_dict
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from typing import Union, Type, Optional
from gym.wrappers import TimeLimit

from stable_baselines3.common.type_aliases import GymEnv
policy_dict = make_policy_dict()
env_tag = 'reach'
gt_policy = policy_dict[env_tag]
pape = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_tag][1]]()
pape._freeze_rand_vec = False
timelimit = TimeLimit(env=pape, max_episode_steps=100)
dv1 = DummyVecEnv([lambda: RolloutInfoWrapper(timelimit)])
class ImitationLearningWrapper:
    def __init__(self, policy, env:GymEnv):
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
    result =  not th.equal(current_obs.reshape(-1)[-3:], check_obsvs.reshape(-1)[-3:])
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
        rollout.make_sample_until(min_timesteps=None, min_episodes=5),
        unwrap=True,
        exclude_infos=False,
    )
    return rollout.flatten_trajectories(rollouts)
transitions = sample_expert_transitions(IGTP)

args_obj = ActiveCriticArgs()
args_obj.set_batchsize(32)
args_obj.set_data_path(path='/home/hendrik/Documents/master_project/LokalData/')
args_obj.set_device(device='cuda')
args_obj.set_feature_extractor(DummyExtractor())
args_obj.set_log_name('reach imitation critic')
args_obj.set_meta_optimizer_lr(1e-2)
args_obj.set_opt_steps(20)
args_obj.set_mlr(5e-5)
args_obj.set_network_setup(NetworkSetup())
args_obj.set_tboard(True)
args_obj.set_demonstrations(demonstrations=transitions)
args_obj.set_n_steps(n_steps=4000)
args_obj.set_epoch_len(epoch_len=100)
args_obj.set_imitation_phase(False)
args_obj.set_weight_decay(1e-2)
args_obj.set_new_epoch(new_epoch=new_epoch)
args_obj.set_eval_epochs(epochs=100)
args_obj.set_complete_modulo(20)

ac = ActiveCritic(
    policy=None,
    env=dv1,
    args_obj=args_obj,
    learning_rate=5e-5,
    extractor=DummyExtractor()
)

train_policy(trainer=ac, 
learn_fct=ac.learn, 
val_env=ac.env, 
logname=args_obj.logname, 
path='/home/hendrik/Documents/master_project/LokalData/TransformerImitationLearning/',
n_epochs=1000,
n_steps=1,
eval_epochs=1,
step_fct=lambda i:i+1,
new_epoch=new_epoch,
extractor=DummyExtractor()
)