import higher
import torch
import torch.nn as nn

def loss_fct_proto(inpt, label):
    return ((inpt - label)**2)
    
class SignalModule():
    def __init__(self, model, loss_fct, optimizer):
        self.model = model
        self.loss_fct = loss_fct
        self.optimizer = optimizer

    def forward(self, inpt):
        return self.model.forward(inpt)

class TaylorSignalModule(SignalModule):
    def __init__(self, model, loss_fct, optimizer, label):
        super().__init__(model=model, loss_fct=loss_fct, optimizer=optimizer)
        self.label = label

    def forward(self, inpt):
        in_value = inpt['proto_result']
        return super().forward(in_value)

    def loss_fct_tailor(self, inpt):
        return self.loss_fct(inpt['tailor_result'], self.label)

#signal_module: forward, loss_fct
def meta_optimizer(main_signal, tailor_signals, inpt, label, epoch, debug_second):
    #build environmtn
    higher_main = 0
    higher_tailor = []
    with higher.innerloop_ctx(main_signal.model, main_signal.optimizer) as (main_fmodel, main_foptim):
        fMainSignal = SignalModule(main_fmodel, main_signal.loss_fct, main_foptim)
        higher_main = fMainSignal
        for ts in tailor_signals:
            with higher.innerloop_ctx(ts.model, ts.optimizer) as (ts_fmodel, ts_fopt):
                fTailorSignal = TaylorSignalModule(ts_fmodel, ts.loss_fct, ts_fopt, ts.label)
                higher_tailor.append(fTailorSignal)
    
    #first forward pass
    proto_result = higher_main.forward(inpt)
    proto_loss = higher_main.loss_fct(proto_result, label)
    #print(f'proto result: {proto_result}')
    
    #compute taior losses
    tailor_losses = []
    for higher_tailor_sig in higher_tailor:
        tailor_inpt = {}
        tailor_inpt['proto_result'] = proto_result
        tailor_result = higher_tailor_sig.forward(tailor_inpt)
        lfi = {}
        lfi['tailor_result'] = tailor_result
        tailor_loss = higher_tailor_sig.loss_fct_tailor(lfi)
        #print(f'tailor loss: {tailor_loss}')
        #optimize main by tailor loss
        higher_main.optimizer.step(tailor_loss)
        tailor_losses.append(tailor_loss)

    #second forward pass
    result = higher_main.model(inpt)
    #print(f'result: {result}')
    loss = higher_main.loss_fct(result, label)
    if epoch%10 == 0:
        print('__________________________________________')
        print(f'loss: {loss}')
        print(f'proto_loss: {proto_loss}')
        for kldasd in tailor_losses:
            print(f'tailor loss: {kldasd}')

    #optimize main and tailor by loss
    higher_main.optimizer.step(loss)

    for i, higher_tailor_sig in enumerate(higher_tailor):
        #print('before')
        #print(list(higher_tailor_sig.model.parameters()))
        higher_tailor_sig.optimizer.step(loss + tailor_losses[i])
        #print(higher_tailor_sig.optimizer.step(tailor_losses[i]))

    if not debug_second:
        main_signal.model.load_state_dict(higher_main.model.state_dict())
    for i, ts in enumerate(tailor_signals):
        ts.model.load_state_dict(higher_tailor[i].model.state_dict())

    return main_signal, tailor_signals

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
    

if __name__ == '__main__':
    torch.manual_seed(0)

    MM = LinearNN([1,100,100,1]).to('cuda')
    MOP = torch.optim.SGD(MM.parameters(), lr = 1e-2)
    MSM = SignalModule(model=MM, loss_fct=loss_fct_proto, optimizer=MOP)

    MMT = LinearNN([1,100,100,1]).to('cuda')
    MOPT = torch.optim.SGD(MMT.parameters(), lr = 1e-2)
    MSMT = [TaylorSignalModule(model=MMT, loss_fct=loss_fct_proto, optimizer=MOPT, label=1)]
    for i in range(11):
        MSM, MSMT = meta_optimizer(MSM, MSMT, torch.ones(1).to('cuda'), torch.zeros(1).to('cuda'), i, debug_second=False)
        #MSM, MSMT = meta_optimizer(MSM, MSMT, torch.ones(1).to('cuda'), torch.zeros(1).to('cuda'), i, debug_second=False)
        #print(f'___________________________________')