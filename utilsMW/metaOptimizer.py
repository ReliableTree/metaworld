from sys import prefix
from time import perf_counter
from turtle import forward

import higher
import torch
import torch.nn as nn
import copy

def loss_fct_proto(inpt, label):
    return ((inpt - label)**2)

class LinearNN(nn.Module):
    def __init__(self, arc):
        super().__init__()
        model = []
        for i in range(len(arc) - 2):
            model.append(nn.Linear(arc[i], arc[i+1]))
            model.append(nn.ReLU())
        model.append(nn.Linear(arc[-2], arc[-1]))
        self.model = nn.Sequential(*model)
        
    def forward(self, inpt):
        return self.model(inpt)
    
class SignalModule():
    def __init__(self, model, loss_fct):
        self.model = model
        self.loss_fct = loss_fct

    def forward(self, inpt):
        return self.model.forward(inpt)

class TaylorSignalModule(SignalModule):
    def __init__(self, model, loss_fct, lr, mlr):
        super().__init__(model=model, loss_fct=loss_fct)
        #self.meta_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.meta_optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=1e-1)
        self.mlr = mlr
        self.lr = lr

    def init_model(self, inpt):
        self.model.super_init = False
        self.model.forward(inpt)
        #self.meta_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.meta_optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=1e-1)

    def forward(self, inpt):
        trajectories = inpt['result']
        original_trajectories = inpt['original']
        inpt_obs = inpt['inpt'][:,:1]
        inpt_obs = inpt_obs.repeat((1, trajectories.size(1), 1))
        #inpt_super = torch.concat((trajectories, original_trajectories, inpt_obs), dim = -1)
        inpt_super = torch.concat((trajectories, inpt_obs), dim = -1)
        result =  super().forward(inpt_super)

        self.lr = self.meta_optimizer.param_groups[0]['lr']
        return result

    def loss_fct_tailor(self, inpt, label):
        #label = 1 means success
        label = label.reshape(-1).type(torch.long)
        label_one_hot = torch.nn.functional.one_hot(label, num_classes = 2)

        inpt = inpt['tailor_result']
        loss_negative = ((inpt[label==0] - label_one_hot[label==0])**2).mean()
        loss_positive = ((inpt[label==1] - label_one_hot[label==1])**2).mean()
        loss = ((inpt.reshape(-1)-label_one_hot.reshape(-1))**2).mean()
        return loss, loss_positive, loss_negative

class MetaModule():
    def __init__(self, main_signal, tailor_signals, lr, return_mode=0, writer=None, device='cuda'):
        self.main_signal = main_signal
        self.tailor_signals = tailor_signals
        self.lr = lr
        self.return_mode = return_mode
        self.writer = writer
        self.optim_run = 0
        self.max_steps =5
        self.last_update = 0
        self.device = device

    def eval(self):
        for ts in self.tailor_signals:
            ts.model.eval()

    def train(self):
        for ts in self.tailor_signals:
            ts.model.train()
    
    def forward(self, inpt, epochs = 100):
        main_signal_state_dict= copy.deepcopy(self.main_signal.model.state_dict())
        gen_result = self.main_signal.forward(inpt)['gen_trj']
        if self.return_mode == 0:
            return {'gen_trj': gen_result, 'inpt_trj' : gen_result}
        elif self.return_mode == 1:
            #opt_inpt = torch.clone(inpt.detach())
            opt_gen_result = torch.clone(gen_result.detach())
            opt_gen_result.requires_grad_(True)
            #optimizer =  torch.optim.Adam(self.main_signal.model.parameters(), lr=self.lr)
            #optimizer = torch.optim.SGD(self.main_signal.model.parameters(), lr=self.lr)
            #optimizer = torch.optim.SGD([opt_gen_result], lr=self.lr)
            optimizer = torch.optim.Adam([opt_gen_result], lr=self.lr)
            #optimizer = torch.optim.AdamW(self.main_signal.model.parameters(), lr=self.lr)

            best_expected_success = None
            best_expected_mean = torch.tensor(float('inf'))
            best_trj = torch.clone(gen_result)
            #for i in range(epochs):
            step = 0
            for ts in self.tailor_signals:
                ts.model.eval()
            gen_result = gen_result.detach()
            '''if (self.optim_run+1) % 10 ==0 and self.optim_run != self.last_update:
                self.last_update = self.optim_run
                self.max_steps *= 1.05'''
            while best_expected_mean > -torch.log(torch.tensor(0.95)) and (step <= self.max_steps):
                optimizer.zero_grad()
                tailor_inpt = {'result':opt_gen_result, 'inpt':inpt, 'original':gen_result}
                tailor_results = []
                for ts in self.tailor_signals:
                    tailor_results.append(ts.forward(tailor_inpt))
                #tailor_result = self.tailor_signal.forward(tailor_inpt)
                expected_succes_before = torch.ones_like(tailor_results[0].max(dim=-1)[1])
                goal_label = torch.ones_like(tailor_results[0][:,0])
                tailor_loss = torch.zeros(1, device=goal_label.device, requires_grad=True)
                for i, tr in enumerate(tailor_results):
                    expected_succes_before = expected_succes_before * tr.max(dim=-1)[1]
                    tailor_loss_inpt = {'tailor_result': tr}
                    tailor_loss = tailor_loss+self.tailor_signals[i].loss_fct_tailor(inpt=tailor_loss_inpt, label=goal_label)[0]

                tailor_loss = tailor_loss/len(self.tailor_signals)
                expected_succes_before = expected_succes_before.type(torch.float).mean()
                #expected_succes_before = tailor_result.max(dim=-1)[1].type(torch.float).mean()
                
                #tailor_loss = tailor_loss + change_trj_loss
                tailor_loss.backward()
                optimizer.step()
                #opt_gen_result = self.main_signal.forward(inpt)['gen_trj']
                tailor_after_inpt = {'result':opt_gen_result, 'inpt':inpt, 'original':gen_result}
                neg_log_tailor_results_after = torch.zeros(inpt.size(0), device=expected_succes_before.device)
                for i, ts in enumerate(self.tailor_signals):
                    neg_log_tailor_results_after = neg_log_tailor_results_after-torch.log(ts.forward(tailor_after_inpt)[:,1])
                #expected_succes_after = tailor_result_after[:,1].mean()
                #expected_succes_after = tailor_result_after.max(dim=-1)[1].type(torch.float).mean()
                if best_expected_success is None:
                    best_expected_success = torch.clone(neg_log_tailor_results_after)
                    best_trj = opt_gen_result.detach()
                else:
                    improve_mask = neg_log_tailor_results_after < best_expected_success
                    best_expected_success[improve_mask]= neg_log_tailor_results_after[improve_mask].detach()
                    best_trj[improve_mask] = opt_gen_result[improve_mask].detach()
                best_expected_mean = best_expected_success.mean()
                self.writer({str(self.optim_run) +' in optimisation ':torch.exp(-best_expected_mean)}, train=False, step=step)
                step += 1
            self.main_signal.model.load_state_dict(main_signal_state_dict)
            return {
                'gen_trj': best_trj.detach(),
                'inpt_trj' : gen_result.detach(),
                'exp_succ_bef': expected_succes_before,
                'exp_succ_after': best_expected_mean,
            }

def meta_optimizer(main_module, tailor_modules, inpt, d_out, epoch, debug_second, force_tailor_improvement, model_params):
    #build environmtn
    proto_loss_dt = 0
    loss_dt = 1
    first_run = True
    inner_loop = 0
    while (proto_loss_dt < loss_dt) and first_run and inner_loop < 10:
        inner_loop += 1
        first_run = force_tailor_improvement
        higher_main = 0
        higher_tailor = []
        main_module.optimizer.zero_grad()
        for tailor_module in tailor_modules:
            tailor_module.optimizer.zero_grad()
            tailor_module.meta_optimizer.zero_grad()
        with higher.innerloop_ctx(main_module.model, main_module.optimizer) as (main_fmodel, main_foptim):
            fMainSignal = SignalModule(main_fmodel, main_module.loss_fct, main_foptim)
            higher_main = fMainSignal
            for ts in tailor_modules:
                ts.optimizer.zero_grad()
                with higher.innerloop_ctx(ts.model, ts.optimizer) as (ts_fmodel, ts_fopt):
                    fTailorSignal = TaylorSignalModule(model=ts_fmodel, loss_fct=ts.loss_fct, optimizer=ts_fopt, meta_optimizer=ts.meta_optimizer)
                    higher_tailor.append(fTailorSignal)
        
        #first forward pass
        proto_result = higher_main.forward(inpt)
        proto_loss, debug_dict = higher_main.loss_fct(d_out = d_out, result = proto_result, model_params=model_params, prefix='proto_')
        proto_loss_dt = proto_loss.detach()
        #print(f'proto result: {proto_result}')
        
        #compute taior losses
        tailor_losses = []
        for higher_tailor_sig in higher_tailor:
            tailor_inpt = {}
            tailor_inpt['result'] = proto_result['gen_trj']
            tailor_inpt['inpt'] = inpt
            tailor_result = higher_tailor_sig.forward(tailor_inpt)
            tailor_loss_input = {}
            tailor_loss_input['tailor_result'] = tailor_result
            tailor_loss = higher_tailor_sig.loss_fct_tailor(inpt = tailor_loss_input, label = torch.ones_like(tailor_result))
            debug_dict['tailor_loss'] = tailor_loss
            #print(f'tailor loss: {tailor_loss}')
            #optimize main by tailor loss
            higher_main.optimizer.step(tailor_loss)
            tailor_losses.append(tailor_loss)
            #print(f'in tailor loss: {tailor_loss}')

        #second forward pass
        result = higher_main.model(inpt)
        #print(f'result shape: {result["gen_trj"].shape}')
        #print(f'result: {result}')
        loss, debug_dict_main = higher_main.loss_fct(d_out = d_out, result=result, model_params=model_params)
        debug_dict['meta_diff'] = proto_loss-loss
        debug_dict.update(debug_dict_main)
        loss_dt = loss.detach()

        #optimize main and tailor by loss

        for i, higher_tailor_sig in enumerate(higher_tailor):
            higher_tailor_sig.optimizer.step(loss)
        for i, ts in enumerate(tailor_modules):
            ts.model.load_state_dict(higher_tailor[i].model.state_dict())
    debug_dict['inner_loop'] = torch.tensor(inner_loop)
    higher_main.optimizer.step(loss)
    if not debug_second:
        main_module.model.load_state_dict(higher_main.model.state_dict())
    for i, ts in enumerate(tailor_modules):
        ts.model.load_state_dict(higher_tailor[i].model.state_dict())

    return main_module, tailor_modules, result, debug_dict

def tailor_optimizer(tailor_modules, succ, failed):
    debug_dict = {}
    if len(succ[0].shape) < 3:
        for ob in succ:
            ob = ob.unsqueeze(0)
        for ob in failed:
            ob = ob.unsqueeze(0)
    s_trj, s_obs, success, s_ftrj = succ
    f_trj, f_obs, fail, f_ftrj = failed

    trajectories = torch.cat((s_trj, f_trj), dim=0)
    ftrj = torch.cat((s_ftrj, f_ftrj), dim=0)
    inpt = torch.cat((s_obs, f_obs), dim=0)
    label = torch.cat((success, fail), dim=0)
    tailor_results = []
    for i, tailor_module in enumerate(tailor_modules):
        tailor_module.meta_optimizer.zero_grad()
        tailor_inpt = {}
        tailor_inpt['result'] = trajectories
        tailor_inpt['inpt'] = inpt
        tailor_inpt['original'] = ftrj
        tailor_result = tailor_module.forward(tailor_inpt)
        tailor_dist = torch.zeros(1, device=tailor_result.device, requires_grad=True)
        for tr in tailor_results:
            tailor_dist = tailor_dist-((tailor_result - tr)**2).mean()
        tailor_dist = tailor_dist/(len(tailor_results)+1)
        tailor_loss_input = {}
        tailor_loss_input['tailor_result'] = tailor_result
        tailor_loss, loss_positive, loss_negative = tailor_module.loss_fct_tailor(inpt = tailor_loss_input, label = label)
        if not torch.isnan(tailor_loss):
            tailor_results.append(torch.clone(tailor_result.detach()))
            debug_dict['tailor dist ' +str(i)] = tailor_dist.mean()
            tailor_loss_dist = (tailor_loss + 10*tailor_dist).mean()
            tailor_loss_dist.backward()
            tailor_module.meta_optimizer.step()
            debug_dict['tailor loss '+str(i)] = tailor_loss.detach()
            debug_dict['tailor loss positive '+str(i)] = loss_positive.detach()
            debug_dict['tailor loss negative '+str(i)] = loss_negative.detach()

            add_max_val_to_dict(debug_dict, 'tailor loss', tailor_loss.detach(), tm = i)
            add_max_val_to_dict(debug_dict, 'tailor loss positive', loss_positive.detach(), tm = i)
            add_max_val_to_dict(debug_dict, 'tailor loss negative', loss_negative.detach(), tm = i)
    return debug_dict

def add_max_val_to_dict(dict, key, val, tm):
    if key in dict:
        dict[key] = torch.max(dict[key], val)
        if val > dict[key]:
            dict[key + ' max'] = torch.tensor(tm)
    else:
        dict[key] = val


if __name__ == '__main__':
    torch.manual_seed(1)

    MM = LinearNN([1,100,100,1]).to('cuda')
    MOP = torch.optim.Adam(params=MM.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2) 
    MSM = SignalModule(model=MM, loss_fct=loss_fct_proto, optimizer=MOP)

    MMT = LinearNN([1,100,100,1]).to('cuda')
    MOPT = torch.optim.Adam(MMT.parameters(), lr = 1e-3, betas=(0.9, 0.999), weight_decay=1e-2) 
    MMOPT = torch.optim.Adam(MMT.parameters(), lr = 1e-3, betas=(0.9, 0.999), weight_decay=1e-2) 
    MSMT = [TaylorSignalModule(model=MMT, loss_fct=loss_fct_proto, optimizer=MOPT, meta_optimizer=MMOPT)]
    label = torch.ones(1).to('cuda')
    inpt = 2*torch.ones(1).to('cuda')
    meta_label = torch.zeros(1).to('cuda')
    for i in range(10):
        MSM, MSMT, result = meta_optimizer(MSM, MSMT, inpt=inpt, label=label, epoch=i, debug_second=False, force_tailor_improvement=True)
        #MSM, MSMT, result = meta_optimizer(MSM, MSMT, inpt=inpt, label=label, epoch=i, debug_second=False)

        #MSM, MSMT, result = meta_optimizer(MSM, MSMT, inpt=inpt, label=label, epoch=i, debug_second=False)

        #tailor_optimizer(MSMT, result, inpt_task=inpt, label=meta_label, epoch=i)
        print(f'___________________________________')