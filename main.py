import torch
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torch.nn as nn
import numpy as np
from models import small,Large
from train import train 
from utils import load_model_weight
from matplotlib import pyplot as plt

#data loading
mnist_trainset = datasets.MNIST(root='/Users/joe/Documents/mnist_data/data', train=True, download=True, transform=tf.ToTensor())
down_sampled_train = torch.from_numpy(np.load('/Users/joe/Documents/main_branch/down_sampled_train.npy'))
concat_train = torch.from_numpy(np.load('/Users/joe/Documents/main_branch/concat_train.npy'))

def cycle_train(epoch1,epoch2,epoch3,cycle):
    '''
    complete one training cycle
    small network -> large network -> small network 
    '''		
    model1 = small()
    model2 = Large()
    
    load_model_weight(model1,model2,small_to_big=False)
    loss = nn.NLLLoss(reduction='none')#nn.MSELoss()#
    optim = torch.optim.Adam(model1.parameters(), lr=0.0001)
    
    position = []
    l,r = 0,50
        
    
    res = train(model1,loss,optim,down_sampled_train,mnist_trainset.targets,epoch1,128)
    position  = np.arange(l,r)
    plt.plot(position,res,'b')
    l +=50
    r +=50
    
    for i in range(cycle):
        model2 = Large()
        if i == 0:
            first = True 
        else:
            first = False
        load_model_weight(model1,model2,small_to_big=True,first=first)
        optim = torch.optim.Adam(model2.parameters(), lr=0.001)
        res = train(model2,loss,optim,concat_train,mnist_trainset.targets,epoch2,128)
        position  = np.arange(l,r)
        plt.plot(position,res,'g')
        l +=50
        r +=50

        model1 = small()
        load_model_weight(model2,model1,small_to_big=False)
        loss = nn.NLLLoss(reduction='none')
        optim = torch.optim.Adam(model1.parameters(), lr=0.0001)
        res = train(model1,loss,optim,down_sampled_train,mnist_trainset.targets,epoch3,128)
        position = np.arange(l,r)
        plt.plot(position,res,'b')
        l +=50
        r +=50
        
    plt.show()

cycle_train(50,50,50,cycle=3)

