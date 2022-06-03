import torch
import torch.nn as nn
import torch.nn.functional as F

class Large(torch.nn.Module):
  '''
  take in concat(full size differene image(28x28), downsampled image(14x14))
  '''
  def __init__(self):
    super(Large, self).__init__()
    self.l1 = nn.Linear(980, 256, bias=False) #980-256
    self.l2 = nn.Linear(256, 20, bias=False)
    self.l3 = nn.Linear(20,10,bias=False)
    self.sm = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    x = self.l3(x)
    x = self.sm(x)
    return x

class small(torch.nn.Module):
  '''
  takes in down sampled image(14x14)
  '''
  def __init__(self):
    super(small, self).__init__()
    self.l1 = nn.Linear(14*14, 128, bias=False)  #196 - 128
    self.l2 = nn.Linear(128, 20, bias=False)
    self.l3 = nn.Linear(20,10,bias=False)
    self.sm = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    x = self.l3(x)
    x = self.sm(x)
    return x
