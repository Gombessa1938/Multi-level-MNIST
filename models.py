import torch
import torch.nn as nn
import config 
from utils import make_layer_input_list
torch.manual_seed(42)
layer_large,layer_medium,layer_small = make_layer_input_list(config)


class Large(torch.nn.Module):
  def __init__(self,image_large_size=64*48+32*24+16*16,coarse_size=1024,l=layer_large):
    '''
    64*48 + 32*24 + 16*16
    '''
    super(Large,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(image_large_size,coarse_size,bias=False))
    for i in range(len(l)-1):
      self.linears.append(nn.Linear(l[i],l[i+1],bias=False))
    self.linears.append(nn.Linear(l[-1],10,bias=False))
    self.ReLU = nn.ReLU()
    self.sm = nn.LogSoftmax(dim=1)#nn.Softmax()
  def forward(self,x):
    for layers in self.linears:
      x = self.ReLU(layers(x))
    x = self.sm(x)
    return x
    
    
class medium(torch.nn.Module):
  def __init__(self,coarse_size=32*24+16*16,l=[512,64,32,16]):
    '''
    32*24 + 16*16
    '''
    super(medium,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(coarse_size,coarse_size//2,bias=False))
    for i in range(len(l)-1):
      self.linears.append(nn.Linear(l[i],l[i+1],bias=False))
    self.linears.append(nn.Linear(l[-1],10,bias=False))
    self.ReLU = nn.ReLU()
    self.sm = nn.LogSoftmax(dim=1)#nn.Softmax()
  def forward(self,x):
    for layers in self.linears:
      x = self.ReLU(layers(x))
    x = self.sm(x)
    return x
  
  
class small(torch.nn.Module):
  def __init__(self,coarse_size=256,l=[128,32,16,16]): 
    '''
    16x16
    '''
    super(small,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(coarse_size,coarse_size//2,bias=False))
    for i in range(len(l)-1):
      self.linears.append(nn.Linear(l[i],l[i+1],bias=False))
    self.linears.append(nn.Linear(l[-1],10,bias=False))
    self.ReLU = nn.ReLU()
    self.sm = nn.LogSoftmax(dim=1)#nn.Softmax()
  def forward(self,x):
    for layers in self.linears:
      x = self.ReLU(layers(x))
    x = self.sm(x)
    return x
  
