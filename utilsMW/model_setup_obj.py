from pyclbr import Function
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional, Union

from PyTorchRL.utils import torch
from LanguagePolicies.model_src.modelTorch import WholeSequenceActor, WholeSequenceCritic, WholeSequenceModel


class NetworkSetup:
    def __init__(self) -> None:
        self.actor_nn = ModelSetup()
        self.critic_nn = ModelSetup()

        self.critic_nn.use_layernorm = False
        self.critic_nn.upconv = False
        self.critic_nn.num_upconvs = 4
        self.critic_nn.stride = 3
        self.critic_nn.d_output = 5
        self.critic_nn.d_model = 60
        self.critic_nn.nhead = 6
        self.critic_nn.d_hid = 60
        self.critic_nn.nlayers = 4
        self.critic_nn.d_result = 1
        self.critic_nn.seq_len = 100
        self.critic_nn.optimizer_class = torch.optim.Adam
        self.model_class:WholeSequenceModel = WholeSequenceCritic
        


class ModelSetup:
    def __init__(self) -> None:
        self.use_layernorm = False
        self.upconv = True
        self.num_upconvs = 5
        self.stride = 3
        self.d_output = 4
        self.nhead = 4
        self.d_hid = 512
        self.d_model = 512
        self.nlayers = 4
        self.seq_len = 100
        self.dilation = 2
        self.d_result = None
        self.ntoken = -1
        self.dropout = 0.2
        self.lr = None
        self.device = 'cuda'
        self.optimizer_class = torch.optim.AdamW
        self.optimizer_kwargs = {}
        self.model_class:WholeSequenceModel = WholeSequenceActor
        
class ActiveCriticArgs:
    def __init__(self) -> None:
        pass

    def set_quick_eval_epochs(self, quick_eval_epochs:int):
        self.quick_eval_epochs = quick_eval_epochs
    
    def set_network_setup(self, network_setup:NetworkSetup):
        self.network_setup = network_setup

    def set_epoch_len(self, epoch_len:int):
        self.network_setup.critic_nn.seq_len = epoch_len
        self.network_setup.actor_nn.seq_len = epoch_len
        self.epoch_len = epoch_len

    def set_new_epoch(self, new_epoch:Function):
        self.new_epoch = new_epoch

    def set_feature_extractor(self, extractor:BaseFeaturesExtractor):
        self.extractor = extractor

    def set_data_path(self, path:str, model_path:Optional[str] = None):
        self.data_path = path
        if model_path is not None:
            self.model_path = path + model_path
        else:
            self.model_path = None

    def set_log_name(self, logname:str):
        self.logname = logname

    def set_mlr(self, mlr:float):
        self.mlr = mlr

    def set_meta_optimizer_lr(self, mo_lr:float):
        self.network_setup.critic_nn.lr = mo_lr

    def set_lr(self, lr:float):
        self.network_setup.actor_nn.lr = lr

    def set_demonstrations(self, demonstrations:list):
        self.demonstrations = demonstrations

    def set_device(self, device:str):
        self.device = device
        self.network_setup.actor_nn.device = device
        self.network_setup.critic_nn.device = device

    def set_tboard(self, tboard:bool):
        self.tboard = tboard

    def set_batchsize(self, batch_size:int):
        self.batch_size = batch_size

    def set_n_steps(self, n_steps:int):
        self.n_steps = n_steps

    def set_imitation_phase(self, imitation_phase:bool):
        self.imitation_phase=imitation_phase

    def set_weight_decay(self, weight_decay:float):
        self.weight_decay = weight_decay

    def set_eval_epochs(self, epochs:int):
        self.eval_epochs = epochs

    def set_opt_steps(self, opt_steps:int):
        self.opt_steps = opt_steps

    def set_complete_modulo(self, complete:int):
        self.complete_modulo = complete

    def set_observable(self, observable:bool):
        self.observable = observable