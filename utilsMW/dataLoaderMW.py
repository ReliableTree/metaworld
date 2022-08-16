from os import path
import torch
from torch.utils.data import DataLoader

class TorchDatasetMW(torch.utils.data.Dataset):
    def __init__(self, path = None, device = 'cpu', n=1):
        self.device = device
        if path is not None:
            path_data = path + 'training_data'
            path_label = path + 'training_label'
            path_phase = path + 'training_reward'
            self.data = torch.load(path_data).to(torch.float32).to(device)[-n:]
            self.label = torch.load(path_label).to(torch.float32).to(device)[-n:]
            self.phase = torch.load(path_phase).to(torch.float32).to(device)[-n:]


    def __len__(self):
            'Denotes the total number of samples'
            return len(self.data)

    def set_data(self, data, label):
        if data.size(1) == 1:
            data = data.repeat([1,self.data.size(1), 1])
        self.data = data.to(self.device)
        self.label = label.to(self.device)
        print(f'train self.data: {self.data.shape}')
        print(f'train self.label: {self.label.shape}')

    def add_data(self, data, label):
        if data.size(1) == 1:
            data = data.repeat([1,self.data.size(1), 1])

        self.data = torch.cat((self.data, data.to(self.device)), dim=0)
        self.label = torch.cat((self.label, label.to(self.device)), dim=0)
        print(f'train self.data: {self.data.shape}')
        print(f'train self.label: {self.label.shape}')

    def __getitem__(self, index):
            'Generates one sample of data'
            return self.data[index], self.label[index]

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

class TorchDatasetTailor(torch.utils.data.Dataset):
    def __init__(self, trajectories, obsv, success:torch.Tensor, ftrj) -> None:
        super().__init__()
        success = success.type(torch.bool)
        self.s_trajectories = trajectories[success==1]
        self.s_ftrj = ftrj[success==1]
        self.s_obsv = obsv[success==1]
        self.success = success[success==1]

        self.f_trajectories = trajectories[success==0]
        self.f_ftrj = ftrj[success==0]
        self.f_obsv = obsv[success==0]
        self.fail = success[success==0]

        self.s_len = len(self.success)
        self.f_len = len(self.fail)
        self.len = max(self.s_len, self.f_len)

    def add_data(self, trajectories, obsv, success, ftrj):
        success = success.type(torch.bool)

        self.s_trajectories = torch.cat((self.s_trajectories, trajectories[success==1]), dim=0)
        self.s_ftrj = torch.cat((self.s_ftrj, ftrj[success==1]), dim=0)
        self.s_obsv = torch.cat((self.s_obsv, obsv[success==1]), dim=0)
        self.success = torch.cat((self.success, success[success==1]), dim=0)

        self.f_trajectories = torch.cat((self.f_trajectories, trajectories[success==0]), dim=0)
        self.f_ftrj = torch.cat((self.f_ftrj, ftrj[success==0]), dim=0)
        self.f_obsv = torch.cat((self.f_obsv, obsv[success==0]), dim=0)
        self.fail = torch.cat((self.fail, success[success==0]), dim=0)

        self.s_len = len(self.success)
        self.f_len = len(self.fail)
        self.len = max(self.s_len, self.f_len)

    def __len__(self):
        return self.len

    def _num_elements(self):
        return self.s_len + self.f_len

    def __getitem__(self, index):
        return (self.s_trajectories[index%self.s_len], self.s_obsv[index%self.s_len], self.success[index%self.s_len], self.s_ftrj[index%self.s_len]),\
             (self.f_trajectories[index%self.f_len], self.f_obsv[index%self.f_len], self.fail[index%self.f_len], self.f_ftrj[index%self.f_len])

if __name__ == '__main__':
    ptd = '/home/hendrik/Documents/master_project/LokalData/metaworld/pick-place/training_data/'
    TD = TorchDatasetMW(path=ptd)
    train_loader = DataLoader(TD, batch_size=16, shuffle=True)
    print(len(train_loader))
    for step, (d, l) in enumerate(train_loader):
        print(d.shape)
        print(l.shape)

        break

