import torch
import torch.nn as nn
import torch.nn.functional as F

class Large(torch.nn.Module):
  '''
  take in concat(full size differene image(32x32), downsampled image(16x16))

  '''
  def __init__(self):
    super(Large, self).__init__()
    self.l1 = nn.Linear(1024+256,256,bias=False)
    self.l2 = nn.Linear(256, 128, bias=False)
    self.l3 = nn.Linear(128, 64, bias=False)
    self.l4 = nn.Linear(64, 32, bias=False)
    self.l5 = nn.Linear(32, 16, bias=False)
    self.l6 = nn.Linear(16, 1, bias=False)
    self.ReLU = nn.ReLU()
  def forward(self, x):
    x = self.ReLU(self.l1(x))
    x = self.ReLU(self.l2(x))
    x = self.ReLU(self.l3(x))
    x = self.ReLU(self.l4(x))
    x = self.ReLU(self.l5(x))
    x = self.ReLU(self.l6(x))
    return x

class small(torch.nn.Module):
  '''
  takes in down sampled image(16x16)
  '''
  def __init__(self):
    super(small, self).__init__()
    self.l1 = nn.Linear(256, 128, bias=False)
    self.l2 = nn.Linear(128, 64, bias=False)
    self.l3 = nn.Linear(64, 32, bias=False)
    self.l4 = nn.Linear(32, 16, bias=False)
    self.l5 = nn.Linear(16, 1, bias=False)
    self.ReLU = nn.ReLU()
  def forward(self, x):
    x = self.ReLU(self.l1(x))
    x = self.ReLU(self.l2(x))
    x = self.ReLU(self.l3(x))
    x = self.ReLU(self.l4(x))
    x = self.ReLU(self.l5(x))
    return x