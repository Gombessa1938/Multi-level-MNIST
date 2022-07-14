import torch
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torch.nn as nn
import numpy as np
from models import small,medium, Large
from train import train 
from utils import load_model_weight
import config
from matplotlib import pyplot as plt
from tqdm import trange 


#data loading
data = np.load('/Users/joe/Documents/llnl_branch/llnl.npz')
label = torch.from_numpy(data['Q'].astype('float32'))
down_sampled_train_small = torch.from_numpy(np.load('/Users/joe/Documents/llnl_branch/down_sampled_train_small.npy'))
concat_train_medium = torch.from_numpy(np.load('/Users/joe/Documents/llnl_branch/concat_train_medium.npy'))
print(concat_train_medium.shape)
concat_train_large = torch.from_numpy(np.load('/Users/joe/Documents/llnl_branch/concat_train_large.npy'))
device = config.device
loss = config.loss

def cycle_train(epoch1,epoch2,epoch3,cycle,loss,res):
    '''
    complete one training cycle
    small network -> large network -> small network 
    '''		
    model1 = small()
    model2 = medium()
    model3 = Large()
    bs = 128
    position = []
    l,r = 0,35
    
    res = []
    for i in range(cycle):
        load_model_weight(model1,model2,small_to_big=False)
        optim = torch.optim.Adam(model1.parameters(), lr=0.001)
        res_ = train(model1,loss,optim,down_sampled_train_small,label,epoch1,bs,device,res)
        res += res_
        # position  = np.arange(l,r)
        # plt.plot(position,res,'b')
        # l  = r
        # r +=10
        
        
        model2 = medium()
        load_model_weight(model1,model2,small_to_big=True,first = True)
        optim = torch.optim.Adam(model2.parameters(), lr=0.001)
        res_ = train(model2,loss,optim,concat_train_medium,label,epoch2,bs,device,res)
        res += res_
        # position = np.arange(l,r)
        # plt.plot(position,res,'g')
        # l = r
        # r +=10
        
        model3 = Large()
        load_model_weight(model2,model3,small_to_big=True,first = True)
        optim = torch.optim.Adam(model2.parameters(), lr=0.0001)
        res_ = train(model3,loss,optim,concat_train_large,label,epoch2,bs,device,res)
        res += res_
        # position = np.arange(l,r)
        # plt.plot(position,res,'r')

    return res


for i in trange()



plt.ylabel('loss')
plt.xlabel('iterations')
plt.plot(0,0,'b',label='small network')
plt.plot(0,0,'g', label = 'medium network')
plt.plot(0,0,'r',label='large network')
plt.legend(loc='upper right')
plt.show()

      
result = []
out = cycle_train(35,10,10,cycle=1,loss=loss,res =result)

