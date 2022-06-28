from os import path
import torch
from torch.utils.data import DataLoader

class TorchDatasetMW(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path, device = 'cpu', n=1):
        path_data = path + 'training_data'
        path_label = path + 'training_label'
        path_phase = path + 'training_reward'
        self.data = torch.load(path_data).to(torch.float32).to(device)[:n]
        self.label = torch.load(path_label).to(torch.float32).to(device)[:n]
        self.phase = torch.load(path_phase).to(torch.float32).to(device)[:n]

        #self.data = torch.cat((self.data, self.phase.unsqueeze(-1)), dim=-1)

    def __len__(self):
            'Denotes the total number of samples'
            return len(self.data)

    def add_data(self, data, label):

        if data.size(1) == 1:
            data = data.repeat([1,self.data.size(1), 1])

        self.data = torch.cat((self.data, data), dim=0)
        self.label = torch.cat((self.label, label), dim=0)

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
    def __init__(self, trajectories, obsv, success) -> None:
        super().__init__()

        self.trajectories = trajectories
        self.obsv = obsv
        self.success = success

    def add_data(self, trajectories, obsv, success):

        self.trajectories = torch.cat((self.trajectories, trajectories), dim=0)
        self.obsv = torch.cat((self.obsv, obsv), dim=0)
        self.success = torch.cat((self.success, success), dim=0)
        

    def __len__(self):
        return len(self.success)

    def num_ele(self):
        return len(self.success)

    def __getitem__(self, index):
        return (self.trajectories[index], self.obsv[index], self.success[index])

if __name__ == '__main__':
    ptd = '/home/hendrik/Documents/master_project/LokalData/metaworld/pick-place/training_data/'
    TD = TorchDatasetMW(path=ptd)
    train_loader = DataLoader(TD, batch_size=16, shuffle=True)
    print(len(train_loader))
    for step, (d, l) in enumerate(train_loader):
        print(d.shape)
        print(l.shape)

        break

