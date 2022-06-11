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
   
# class Large(torch.nn.Module):
#   '''
#   take in concat(full size differene image(32x32), downsampled image(16x16))
#   '''
#   def __init__(self,image_large_size=32,num_layers=None):
#     super(Large, self).__init__()
#     self.l1 = nn.Linear(1024+256,256,bias=False)
#     self.l2 = nn.Linear(256, 128, bias=False)
#     self.l3 = nn.Linear(128, 64, bias=False)
#     self.l4 = nn.Linear(64, 16, bias=False)
#     self.l5 = nn.Linear(16, 1, bias=False)
#     self.ReLU = nn.ReLU()
#   def forward(self, x):
#     x = self.ReLU(self.l1(x))
#     x = self.ReLU(self.l2(x))
#     x = self.ReLU(self.l3(x))
#     x = self.ReLU(self.l4(x))
#     x = self.ReLU(self.l5(x))
#     return x

# class small(torch.nn.Module):
#   '''
#   takes in down sampled image(16x16)
#   '''
#   def __init__(self,image_coarse_size=16,num_layers=None):
#     super(small, self).__init__()
#     self.l1 = nn.Linear(256, 128, bias=False)
#     self.l2 = nn.Linear(128, 64, bias=False)
#     self.l3 = nn.Linear(64, 32, bias=False)
#     self.l4 = nn.Linear(32, 16, bias=False)
#     self.l5 = nn.Linear(16, 1, bias=False)
#     self.ReLU = nn.ReLU()
#   def forward(self, x):
#     x = self.ReLU(self.l1(x))
#     x = self.ReLU(self.l2(x))
#     x = self.ReLU(self.l3(x))
#     x = self.ReLU(self.l4(x))
#     x = self.ReLU(self.l5(x))
#     return x