import torch
import torch.nn as nn
import torch.nn.functional as F

class Large(torch.nn.Module):
  '''
  take in concat(full size differene image(28x28), downsampled image(14x14))

  '''
  def __init__(self):
    super(Large, self).__init__()
    self.l1 = nn.Linear(980, 128, bias=False) #28*28 + 14*14  ---> 128, 128 --->10
    self.l2 = nn.Linear(128, 10, bias=False)
    self.sm = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    x = self.sm(x)
    return x

class small(torch.nn.Module):
  '''
  takes in down sampled image(14x14)
  '''
  def __init__(self):
    super(small, self).__init__()
    self.l1 = nn.Linear(14*14, 128, bias=False)  # 196-->128, 128-->10
    self.l2 = nn.Linear(128, 10, bias=False)
    self.sm = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    x = self.sm(x)
    return x