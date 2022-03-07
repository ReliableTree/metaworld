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

