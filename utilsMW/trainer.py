from ast import arg
from cProfile import label
from MetaWorld.utilsMW.model_setup_obj import NetworkSetup
from MetaWorld.searchTest.utils import VecExtractor
from MetaWorld.utilsMW.network_trainer import NetworkTrainer
from MetaWorld.searchTest.utils import parse_sampled_transitions 
from LanguagePolicies.utils.Transformer import TailorTransformer
from LanguagePolicies.model_src.modelTorch import WholeSequenceActor
from MetaWorld.searchTest.utils import count_parameters
from utilsMW.dataLoaderMW import TorchDatasetMW
from torch.utils.data import DataLoader
import torch.nn as nn
from MetaWorld.utilsMW.model_setup_obj import ActiveCriticArgs
import torch


class ActiveCritic(nn.Module):
    def __init__(self, policy, env, args_obj:ActiveCriticArgs, extractor) -> None:
        super().__init__()
        self.env = env
        self.extractor = extractor
        
        actor   = WholeSequenceActor(model_setup=args_obj.network_setup.actor_nn, device=args_obj.device).to(args_obj.device)
        critics=[TailorTransformer(model_setup=args_obj.network_setup.critic_nn) for i in range(1)]
        self.initial_TT_parameters = critics[0].parameters()
        data = TorchDatasetMW(device=args_obj.device)
        data_eval = TorchDatasetMW(device=args_obj.device)

        if args_obj.demonstrations is not None:
            expert_actions, expert_observations, rewards = parse_sampled_transitions(transitions=args_obj.demonstrations, new_epoch=args_obj.new_epoch, extractor=self.extractor)
            data.set_data(inpt=expert_observations, label=expert_actions, success=rewards)
            data_eval.set_data(inpt=expert_observations, label=expert_actions, success=rewards)

        print(f'len(train_data): {len(data)}')
        
        env_tag = 'pickplace'
        #data_path, logname, lr, mlr, mo_lr, gamma_sl = 0.995, device = 'cuda', tboard=True
        
        network = NetworkTrainer(
            actor=actor, 
            critic=critics, 
            env_tag=env_tag, 
            env=self.env, 
            data_path=args_obj.data_path,
            logname=args_obj.logname, 
            mlr=args_obj.mlr, 
            mo_lr=args_obj.meta_optimizer_lr,
            gamma_sl = 1, 
            device=args_obj.device, 
            tboard=args_obj.tboard,
            network_args_obj=args_obj
            )
        network.setDatasets(train_data=data, val_data=data_eval)

        network.setup_model()
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