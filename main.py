import torch
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torch.nn as nn
import numpy as np
from models import small,Large
from train import train 
from utils import load_model_weight
import config
from matplotlib import pyplot as plt

#data loading
data = np.load('/Users/joe/Documents/llnl_branch/llnl.npz')
label = torch.from_numpy(data['Q'].astype('float32'))
down_sampled_train = torch.from_numpy(np.load('/Users/joe/Documents/llnl_branch/down_sampled_train.npy'))
concat_train = torch.from_numpy(np.load('/Users/joe/Documents/llnl_branch/concat_train.npy'))
device = config.device
loss = config.loss

def cycle_train(epoch1,epoch2,epoch3,cycle,loss,res):
    '''
    complete one training cycle
    small network -> large network 
    '''		
    model1 = small()
    model2 = Large()
    position = []
    l,r = 0,100
    
    for i in range(cycle):
        
        load_model_weight(model1,model2,small_to_big=False)
        optim = torch.optim.Adam(model1.parameters(), lr=0.0001)
        res = train(model1,loss,optim,down_sampled_train,label,epoch1,128,device,res)
        position  = np.arange(l,r)
        plt.plot(position,res,'b')
        l  = r
        r +=200

        model2 = Large()
        load_model_weight(model1,model2,small_to_big=True,first = True)
        optim = torch.optim.Adam(model2.parameters(), lr=0.0001)
        res = train(model2,loss,optim,concat_train,label,epoch2,128,device,res)
        
        position = np.arange(l,r)
        plt.plot(position,res,'g')

    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.plot(0,0,'b',label='small network')
    plt.plot(0,0,'g', label = 'medium network')
    #plt.plot(0,0,'r',label='large network')
    plt.legend(loc='upper right')
    plt.show()
    plt.show() 
        
    return res
      
result = []
out = cycle_train(100,200,50,cycle=1,loss=loss,res =result)

