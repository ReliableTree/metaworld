from unittest import result
import torch

def cat_obs_trj(obs, trj):
    if type(trj) is torch.Tensor:
        return torch.cat((obs.repeat(1, trj.size(1), 1), trj), dim=-1)
    else:
        return torch.cat((obs.repeat(1, trj[1], 1), torch.zeros(size=trj, device=obs.device)), dim=-1)

def right_stack_obj_trj(obs, inpt):
    #inpt = N,L,D
    embed_size = obs.size(-1)
    if type(inpt) is torch.Tensor:
        result = inpt
    else:
        result = torch.zeros(size=inpt, device=obs.device)
    result[:,-1,-embed_size:] = obs.squeeze()
    return result