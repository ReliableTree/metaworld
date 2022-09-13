from os import path
import torch
from torch.utils.data import DataLoader

def apply_triu(inpt, diagonal):
    exp_inpt = inpt.unsqueeze(1)
    shape = exp_inpt.shape
    #shape = batch, 1, seq, dims...
    exp_inpt = exp_inpt.repeat([1, shape[2], *[1]*(len(shape)-2)])
    mask = torch.triu(torch.ones([shape[2], shape[2]], device=inpt.device), diagonal=diagonal).T
    #batch, seq, seq, dims...
    exp_out = exp_inpt * mask[None,:,:,None]
    return exp_out

def make_part_obs_data(obs, act, success):
    part_obs = apply_triu(obs, diagonal=0)
    part_obs = part_obs.reshape(-1, part_obs.shape[-2], part_obs.shape[-1])
    part_acts = apply_triu(act, diagonal=1)
    part_acts = part_acts.reshape(-1, part_acts.shape[-2], part_acts.shape[-1])
    inpt = torch.cat((part_obs, part_acts), dim=-1)
    label = act.unsqueeze(1).repeat([1, obs.shape[1], 1, 1])
    label = label.reshape(-1, label.shape[-2], label.shape[-1])
    success = success.unsqueeze(0).repeat([obs.shape[1], 1, 1]).permute(2,0,1)
    success = success.reshape([-1])
    return inpt, label, success


class TorchDatasetMW(torch.utils.data.Dataset):
    def __init__(self, path = None, observable = True, device = 'cpu', n=1):
        self.device = device
        self.inpt = None
        self.label = None
        self.observable = observable
        if path is not None:
            path_data = path + 'training_data'
            path_label = path + 'training_label'
            self.inpt = torch.load(path_data).to(torch.float32).to(device)[-n:]
            self.label = torch.load(path_label).to(torch.float32).to(device)[-n:]
        self.onyl_positiv = None


    def __len__(self):
            'Denotes the total number of samples'
            if self.inpt is not None:
                return len(self.inpt)
            else:
                return 0
    def set_data(self, inpt, label, success):

        print(f'before observations: {inpt.shape}')
        print(f'before actions: {label.shape}')

        if self.observable:
            inpt, label, success = make_part_obs_data(obs=inpt, act=label, success=success)

        print(f'observations: {inpt.shape}')
        print(f'actions: {label.shape}')
        print(f'success; {success.shape}')

        self.success = success.to(self.device)
        self.inpt = inpt.to(self.device)
        self.label = label.to(self.device)
        print(f'train self.data: {self.inpt.shape}')
        print(f'train self.label: {self.label.shape}')

    def add_data(self, inpt, label, success):
        if self.inpt is None:
            self.set_data(inpt, label, success)
        else:
            if self.observable:
                inpt, label, success = make_part_obs_data(obs=inpt, act=label, success=success)
            self.inpt = torch.cat((self.inpt, inpt.to(self.device)), dim=0)
            self.label = torch.cat((self.label, label.to(self.device)), dim=0)
            self.success = torch.cat((self.success, success.to(self.device)), dim=0)
            print(f'train self.data: {self.inpt.shape}')
            print(f'train self.label: {self.label.shape}')

    def __getitem__(self, index):
            assert self.onyl_positiv is not None, 'traindata only positiv not set'
            if self.onyl_positiv:
                return self.inpt[self.success][index], self.label[self.success][index], self.success[self.success][index]
            else:
                return self.inpt[index], self.label[index], self.success[index]



class TorchDatasetMWToy(torch.utils.data.Dataset):
  def __init__(self, path, device = 'cpu'):
      path_data = path + 'inpt'
      path_label = path + 'label'
      self.data = torch.load(path_data).to(torch.float32).to(device)
      self.label = torch.load(path_label).to(torch.float32).to(device)
      

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.data[index], self.label[index]


if __name__ == '__main__':
    ptd = '/home/hendrik/Documents/master_project/LokalData/metaworld/pick-place/training_data/'
    TD = TorchDatasetMW(path=ptd)
    train_loader = DataLoader(TD, batch_size=16, shuffle=True)
    print(len(train_loader))
    for step, (d, l) in enumerate(train_loader):
        print(d.shape)
        print(l.shape)

        break

