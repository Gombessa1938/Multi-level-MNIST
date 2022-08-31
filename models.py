import torch
import torch.nn as nn
import config 
from utils import make_layer_input_list
torch.manual_seed(42)

layer_large,layer_medium,layer_small = make_layer_input_list(config)

class Large(torch.nn.Module):
  def __init__(self,large_size=32*32+16*16+8*8,l=[800,512,256,64]):
    '''
    This is a pytorch network class
    Args:
      large_size: int (default is 32*32 + 16*16 + 8*8)
      l: List this is a list of layer size.
    '''
    super(Large,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(large_size,l[0],bias=False))
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
  def __init__(self,medium_size=16*16+8*8,l=[160,64,32,16]):
    '''
    This is a pytorch network class
    Args:
      medium_size: int (default is 16*16 + 8*8)
      l: List this is a list of layer size.
    '''
    super(medium,self).__init__()
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(medium_size,medium_size//2,bias=False))
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
  def __init__(self,coarse_size = 8*8, l = [32,32,16,16]): 
    '''
    This is a pytorch network class
    Args:
      coarse_size: int (default is 8*8)
      l: List this is a list of layer size.
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
  
