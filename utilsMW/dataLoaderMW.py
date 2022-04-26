from os import path
import torch
from torch.utils.data import DataLoader

class TorchDatasetMW(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path, device = 'cpu'):
      path_data = path + 'training_data'
      path_label = path + 'training_label'
      path_phase = path + 'training_reward'
      self.data = torch.load(path_data).to(torch.float32).to(device)
      self.label = torch.load(path_label).to(torch.float32).to(device)
      self.phase = torch.load(path_phase).to(torch.float32).to(device)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.data[index], (self.label[index], self.phase[index])

if __name__ == '__main__':
    ptd = '/home/hendrik/Documents/master_project/LokalData/metaworld/pick-place/training_data/'
    TD = TorchDatasetMW(path=ptd)
    train_loader = DataLoader(TD, batch_size=16, shuffle=True)
    print(len(train_loader))
    for step, (d, l) in enumerate(train_loader):
        print(d.shape)
        print(l.shape)

        break

class TorchDatasetTailor(torch.utils.data.Dataset):
    def __init__(self, trajectories, obsv, success) -> None:
        super().__init__()
        self.s_trajectories = trajectories[success==1]
        self.s_obsv = obsv[success==1]
        self.success = success[success==1]

        self.f_trajectories = trajectories[success==0]
        self.f_obsv = obsv[success==0]
        self.fail = success[success==0]

        self.s_len = len(self.success)
        self.f_len = len(self.fail)
        self.len = max(self.s_len, self.f_len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.s_trajectories[index%self.s_len], self.s_obsv[index%self.s_len], self.success[index%self.s_len]),\
             (self.f_trajectories[index%self.f_len], self.f_obsv[index%self.f_len], self.fail[index%self.f_len])