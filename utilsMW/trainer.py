from ast import arg
from MetaWorld.utilsMW.model_setup_obj import NetworkSetup
from MetaWorld.searchTest.utils import VecExtractor
from MetaWorld.utilsMW.network_trainer import NetworkTrainer
from MetaWorld.searchTest.utils import parse_sampled_transitions 
from LanguagePolicies.utils.Transformer import TailorTransformer
from LanguagePolicies.model_src.modelTorch import PolicyTranslationModelTorch
from MetaWorld.searchTest.utils import count_parameters
from utilsMW.dataLoaderMW import TorchDatasetMW
from torch.utils.data import DataLoader
import torch.nn as nn
from MetaWorld.utilsMW.model_setup_obj import ActiveCriticArgs


class ActiveCritic(nn.Module):
    def __init__(self, policy, env, args_obj:ActiveCriticArgs, learning_rate, extractor) -> None:
        super().__init__()
        self.env = env
        self.extractor = extractor
        
        actor   = PolicyTranslationModelTorch(od_path="", model_setup=args_obj.network_setup.plan_nn, device=args_obj.device).to(args_obj.device)
        critics=[TailorTransformer(model_setup=args_obj.network_setup.critic_nn) for i in range(1)]
        
        data = TorchDatasetMW(device=args_obj.device)

        expert_actions, expert_observations, rewards = parse_sampled_transitions(transitions=args_obj.demonstrations, new_epoch=args_obj.new_epoch, extractor=self.extractor)
        data.set_data(data=expert_observations, label=expert_actions)
        print(f'len(train_data): {len(data)}')
        train_loader = DataLoader(data, batch_size=args_obj.batch_size, shuffle=True)

        data_eval = TorchDatasetMW(device=args_obj.device)
        data_eval.set_data(data=expert_observations, label=expert_actions)
        eval_loader = DataLoader(data_eval, batch_size=args_obj.batch_size, shuffle=True)
        
        env_tag = 'pickplace'
        #data_path, logname, lr, mlr, mo_lr, gamma_sl = 0.995, device = 'cuda', tboard=True
        
        network = NetworkTrainer(
            model=actor, 
            tailor_models=critics, 
            env_tag=env_tag, 
            env=self.env, 
            data_path=args_obj.data_path,
            logname=args_obj.logname, 
            lr=learning_rate, 
            mlr=args_obj.mlr, 
            mo_lr=args_obj.meta_optimizer_lr,
            gamma_sl = 1, 
            device=args_obj.device, 
            tboard=args_obj.tboard,
            network_args_obj=args_obj
            )
        network.setDatasets(train_loader=train_loader, val_loader=eval_loader)

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