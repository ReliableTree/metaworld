import torch.nn as nn
from LanguagePolicies.model_src.modelTorch import (WholeSequenceActor,
                                                   WholeSequenceCritic)
from MetaWorld.searchTest.utils import parse_sampled_transitions
from MetaWorld.utilsMW.network_trainer import NetworkTrainer

from utilsMW.dataLoaderMW import TorchDatasetMW
from MetaWorld.utilsMW.model_setup_obj import ActiveCriticArgs


class ActiveCritic(nn.Module):
    def __init__(self, policy, env, args_obj:ActiveCriticArgs, extractor) -> None:
        super().__init__()
        self.env = env
        self.extractor = extractor
        
        actor   = WholeSequenceActor(model_setup=args_obj.network_setup.actor_nn).to(args_obj.device)
        critic = WholeSequenceCritic(model_setup=args_obj.network_setup.critic_nn).to(args_obj.device)
        data = TorchDatasetMW(device=args_obj.device)
        data_eval = TorchDatasetMW(device=args_obj.device)

        if args_obj.demonstrations is not None:
            expert_actions, expert_observations, rewards = parse_sampled_transitions(transitions=args_obj.demonstrations, new_epoch=args_obj.new_epoch, extractor=self.extractor)
            data.set_data(inpt=expert_observations, label=expert_actions, success=rewards)
            data_eval.set_data(inpt=expert_observations, label=expert_actions, success=rewards)

        print(f'len(train_data): {len(data)}')
        
        network = NetworkTrainer(
            actor=actor, 
            critic=critic, 
            env=self.env, 
            network_args_obj=args_obj
            )
        network.setDatasets(train_data=data, val_data=data_eval)

        if args_obj.model_path is not None:
            network.loadNetworkFromFile(path=args_obj.model_path)

        self.network = network
        self.policy = network.policy

        self.args_obj = args_obj

    def learn(self, n_epochs):
        self.policy.return_mode = 0
        self.policy.train()
        self.network.train(epochs=n_epochs)

    def eval(self,):
        if self.args_obj.imitation_phase:
            self.policy.return_mode = 0
        else:
            self.policy.return_mode = 1
        self.policy.eval()
