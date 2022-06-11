from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F

class Large(torch.nn.Module):
  def __init__(self,image_large_size=1024,coarse_size=256,l=[256,128,64,16]):
    super(Large,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(image_large_size+coarse_size,coarse_size,bias=False))
    for i in range(len(l)-1):
      self.linears.append(nn.Linear(l[i],l[i+1],bias=False))
    self.linears.append(nn.Linear(l[-1],1,bias=False))
    self.ReLU = nn.ReLU()
    
  def forward(self,x):
    for layers in self.linears:
      x = self.ReLU(layers(x))
    return x
    
    
class small(torch.nn.Module):
  def __init__(self,coarse_size=256,l=[128,64,32,16]):
    super(small,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(coarse_size,coarse_size//2,bias=False))
    for i in range(len(l)-1):
      self.linears.append(nn.Linear(l[i],l[i+1],bias=False))
    self.linears.append(nn.Linear(l[-1],1,bias=False))
    self.ReLU = nn.ReLU()
    
  def forward(self,x):
    for layers in self.linears:
      x = self.ReLU(layers(x))
    return x
  
  
class tiny(torch.nn.Module):
  def __init__(self,coarse_size=64,l=[64,32,16,16]):
    super(tiny,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(coarse_size,coarse_size//2,bias=False))
    for i in range(len(l)-1):
      self.linears.append(nn.Linear(l[i],l[i+1],bias=False))
    self.linears.append(nn.Linear(l[-1],1,bias=False))
    self.ReLU = nn.ReLU()
    
  def forward(self,x):
    for layers in self.linears:
      x = self.ReLU(layers(x))
    return x
   