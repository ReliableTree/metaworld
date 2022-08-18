from email.policy import strict
from turtle import forward
from unittest import result
from tenacity import retry_base
import torch
import copy
from stable_baselines3.common.policies import BaseModel
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from MetaWorld.utilsMW.metaOptimizer import TaylorSignalModule, SignalModule
from MetaWorld.utilsMW.model_setup_obj import ActiveCriticArgs

"""
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[nn.Module] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,

"""

class ActiveCriticPolicy(BaseModel):
    def __init__(
        self, 
        observation_space,
        action_space,
        main_signal, 
        tailor_signals:TaylorSignalModule, 
        lr, 
        return_mode=0, 
        writer=None, 
        args_obj:ActiveCriticArgs = None
        ):

        super().__init__(observation_space, action_space)
        
        self.main_signal = main_signal
        self.tailor_signals:List[TaylorSignalModule] = tailor_signals
        self.lr = lr
        self.return_mode = return_mode
        self.writer = writer
        self.optim_run = 0
        self.max_steps = args_obj.opt_steps
        self.last_update = 0
        self.last_goal = None
        self.seq_len = None
        self.current_step = 0
        self.args_obj = args_obj

    def eval(self):
        for ts in self.tailor_signals:
            ts.model.eval()

    def train(self):
        for ts in self.tailor_signals:
            ts.model.train()

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:

        vec_obsv = self.args_obj.extractor.forward(observation)

        if self.last_goal is not None:
            if self.args_obj.new_epoch(self.last_goal, vec_obsv):
                self.current_step = 0
                #print('new epsiode')
        if self.current_step == 0:

            self.last_goal = vec_obsv
            vec_obsv = vec_obsv.unsqueeze(1).repeat([1,self.args_obj.epoch_len, 1]).to(self.args_obj.device)
            self.pred_actions = self.forward(inpt=vec_obsv)['gen_trj'].detach().cpu().numpy()

            #self.pred_actions = self.forward(inpt=vec_obsv)['gen_trj']
            self.seq_len = self.pred_actions.shape[1]    
        result = self.pred_actions[0, self.current_step]
        if self.current_step == self.seq_len - 1:
            self.current_step = 0
            #print('finished epsiode')
        else:
            self.current_step += 1
        return result, -1

    def forward(self, inpt):
        if type(inpt) is torch.Tensor:
            return self._forward(inpt)
        else:
            return self.predict(observation=inpt)

    
    def _forward(self, inpt):
        main_signal_state_dict= copy.deepcopy(self.main_signal.model.state_dict())
        gen_result = self.main_signal.forward(inpt)['gen_trj']
        if self.return_mode == 0:
            return {'gen_trj': gen_result, 'inpt_trj' : gen_result}
        elif self.return_mode == 1:
            opt_gen_result = torch.clone(gen_result.detach())
            opt_gen_result.requires_grad_(True)
            optimizer = torch.optim.AdamW([opt_gen_result], lr=self.lr)

            best_expected_success = None
            best_expected_mean = torch.tensor(float('inf'))
            best_trj = torch.clone(gen_result)
            step = 0
            if self.tailor_signals[0].init:
                for ts in self.tailor_signals:
                    ts.model.eval()
            gen_result = gen_result.detach()
            while best_expected_mean > -torch.log(torch.tensor(0.95)) and (step <= self.max_steps):
                optimizer.zero_grad()
                tailor_inpt = {'result':opt_gen_result, 'inpt':inpt, 'original':gen_result}
                tailor_results = []
                for ts in self.tailor_signals:
                    tailor_results.append(ts.forward(tailor_inpt))
                expected_succes_before = torch.ones_like(tailor_results[0].max(dim=-1)[1])
                goal_label = torch.ones_like(tailor_results[0][:,0])
                tailor_loss = torch.zeros(1, device=goal_label.device, requires_grad=True)
                for i, tr in enumerate(tailor_results):
                    expected_succes_before = expected_succes_before * tr.max(dim=-1)[1]
                    tailor_loss_inpt = {'tailor_result': tr}
                    tailor_loss = tailor_loss+self.tailor_signals[i].loss_fct_tailor(inpt=tailor_loss_inpt, label=goal_label)[0]

                tailor_loss = tailor_loss/len(self.tailor_signals)
                expected_succes_before = expected_succes_before.type(torch.float).mean()

                tailor_loss.backward()
                optimizer.step()
                tailor_after_inpt = {'result':opt_gen_result, 'inpt':inpt, 'original':gen_result}
                neg_log_tailor_results_after = torch.zeros(inpt.size(0), device=expected_succes_before.device)
                for i, ts in enumerate(self.tailor_signals):
                    neg_log_tailor_results_after = neg_log_tailor_results_after-torch.log(ts.forward(tailor_after_inpt)[:,1])
                if best_expected_success is None:
                    best_expected_success = torch.clone(neg_log_tailor_results_after)
                    best_trj = opt_gen_result.detach()
                else:
                    improve_mask = neg_log_tailor_results_after < best_expected_success
                    best_expected_success[improve_mask]= neg_log_tailor_results_after[improve_mask].detach()
                    best_trj[improve_mask] = opt_gen_result[improve_mask].detach()
                best_expected_mean = best_expected_success.mean()
                self.writer({str(self.optim_run) +' in optimisation ':torch.exp(-best_expected_mean.detach())}, train=False, step=step)
                step += 1
            self.main_signal.model.load_state_dict(main_signal_state_dict, strict=False)
            return {
                'gen_trj': best_trj.detach(),
                'inpt_trj' : gen_result.detach(),
                'exp_succ_bef': expected_succes_before,
                'exp_succ_after': torch.exp(-best_expected_mean.detach()),
            }