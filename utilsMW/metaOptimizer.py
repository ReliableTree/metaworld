from sys import prefix
from time import perf_counter

from numpy import NaN
import higher
import torch
import torch.nn as nn

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
    def __init__(self, model, loss_fct, optimizer):
        self.model = model
        self.loss_fct = loss_fct
        self.optimizer = optimizer

    def forward(self, inpt):
        return self.model.forward(inpt)

class TaylorSignalModule(SignalModule):
    def __init__(self, model, loss_fct, optimizer, meta_optimizer):
        super().__init__(model=model, loss_fct=loss_fct, optimizer=optimizer)
        self.meta_optimizer = meta_optimizer

    def forward(self, inpt):
        trajectories = inpt['result']
        inpt_obs = inpt['inpt'][:,:1]
        inpt_obs = inpt_obs.repeat((1, trajectories.size(1), 1))
        inpt_super = torch.concat((trajectories, inpt_obs), dim = -1)
        #print(f'max {inpt_super.max()}')
        #print(f'min: {inpt_super.min()}')
        #print(f'inpt_super obs shape: {inpt_super.shape}')

        #print(f'invalue: {in_value}')
        #inpt shape: batch, seq, dim
        result =  super().forward(inpt_super)
        return result
    #def signal_fct_tailor(self, inpt):
    #    return self.loss_fct(inpt['tailor_result'], 0)

    def loss_fct_tailor(self, inpt, label):
        #label = 1 means success
        label = label.reshape(-1).type(torch.long)
        label_one_hot = torch.nn.functional.one_hot(label, num_classes = 2)
        inpt = inpt['tailor_result']
        loss_negative = ((inpt[label==0] - label_one_hot[label==0])**2).mean()
        loss_positive = ((inpt[label==1] - label_one_hot[label==1])**2).mean()
        loss = ((inpt.reshape(-1)-label_one_hot.reshape(-1))**2).mean()
        return loss, loss_positive, loss_negative


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
    s_trj, s_obs, success = succ
    f_trj, f_obs, fail = failed
    trajectories = torch.cat((s_trj, f_trj), dim=0)
    inpt = torch.cat((s_obs, f_obs), dim=0)
    label = torch.cat((success, fail), dim=0)
    for tailor_module in tailor_modules:
        tailor_module.meta_optimizer.zero_grad()
        tailor_inpt = {}
        tailor_inpt['result'] = trajectories
        tailor_inpt['inpt'] = inpt
        tailor_result = tailor_module.forward(tailor_inpt)
        tailor_loss_input = {}
        tailor_loss_input['tailor_result'] = tailor_result
        tailor_loss, loss_positive, loss_negative = tailor_module.loss_fct_tailor(inpt = tailor_loss_input, label = label)
        if not torch.isnan(tailor_loss):
            tailor_loss.backward()
            tailor_module.meta_optimizer.step()
            debug_dict['tailor loss'] = tailor_loss.detach()
            debug_dict['tailor loss positive'] = loss_positive.detach()
            debug_dict['tailor loss negative'] = loss_negative.detach()
    return debug_dict


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