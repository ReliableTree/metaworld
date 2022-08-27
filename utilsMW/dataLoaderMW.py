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
        self.success = success.type(torch.bool) 
        self.trajectories = trajectories
        self.obsv = obsv
        self.ftrj = trajectories
        self.len = len(self.success)


    def add_data(self, trajectories, obsv, success, ftrj):
        self.success = torch.cat((success.type(torch.bool), self.success), dim=0)
        self.obsv = torch.cat((obsv, self.obsv), dim=0)
        self.trajectories = torch.cat((trajectories, self.trajectories), dim=0)
        self.ftrj = torch.cat((ftrj, self.ftrj), dim=0)

        self.len = len(self.success)

    def __len__(self):
        return self.len

    def _num_elements(self):
        return self.len

    def __getitem__(self, index):
        return self.trajectories[index], self.obsv[index], self.success[index], self.ftrj[index]

if __name__ == '__main__':
    ptd = '/home/hendrik/Documents/master_project/LokalData/metaworld/pick-place/training_data/'
    TD = TorchDatasetMW(path=ptd)
    train_loader = DataLoader(TD, batch_size=16, shuffle=True)
    print(len(train_loader))
    for step, (d, l) in enumerate(train_loader):
        print(d.shape)
        print(l.shape)

        break

