import torch

def cat_obs_trj(obs, trj):
    if type(trj) is torch.Tensor:
        return torch.cat((obs.repeat(1, trj.size(1), 1), trj), dim=-1)
    else:
        return torch.cat((obs.repeat(1, trj[1], 1), torch.zeros(size=trj, device=obs.device)), dim=-1)