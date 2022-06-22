import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)
class Large(torch.nn.Module):
  def __init__(self,image_large_size=28*28,coarse_size=14*14,l=[14*14,128,64,16]):
    super(Large,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(image_large_size+coarse_size,coarse_size,bias=False))
    for i in range(len(l)-1):
      self.linears.append(nn.Linear(l[i],l[i+1],bias=False))
    self.linears.append(nn.Linear(l[-1],10,bias=False))
    self.ReLU = nn.ReLU()
    self.sm = nn.Softmax(dim=1)#nn.LogSoftmax(dim=1)
  def forward(self,x):
    for i,layers in enumerate(self.linears):
      x = layers(x)
      if i != len(self.linears)-1:
        x = self.ReLU(x)
      else:
        x = self.sm(x)
    return x
    
    
class small(torch.nn.Module):
  def __init__(self,coarse_size=14*14,l=[128,64,32,16]):
    super(small,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(coarse_size,l[0],bias=False))
    for i in range(len(l)-1):
      self.linears.append(nn.Linear(l[i],l[i+1],bias=False))
    self.linears.append(nn.Linear(l[-1],10,bias=False))
    self.ReLU = nn.ReLU()
    self.sm = nn.Softmax(dim=1)#nn.LogSoftmax(dim=1)
  def forward(self,x):
    for i,layers in enumerate(self.linears):
      x = layers(x)
      if i != len(self.linears)-1:
        x = self.ReLU(x)
      else:
        x = self.sm(x)
    return x
  
