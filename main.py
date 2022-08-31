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

mnist_trainset = config.mnist_trainset
label = mnist_trainset.targets
down_sampled_train_small = config.down_sampled_train_small
concat_train_medium = config.concat_train_medium
concat_train_large = config.concat_train_large

device = config.device
loss = config.loss
torch.manual_seed(42)


def cycle_train(epoch1,epoch2,epoch3,cycle,loss,res):
    '''
    complete one training cycle
    small network -> medium network -> large network 
    
    for use more than one cycle, you can do things below
    small network -> medium network -> large network -> medium network ->small network 
    but this requires some simple modification of the code.
    
    Arges:
        epoch 1: int epoch for small network
        epoch 2: int epoch for medium network
        epoch 3: int epoch for large network
        cycle: int number of time you train the network, defult is 1
        loss: pytorch loss function
        res: List list to keep track result.
    Return:
        res: List accuracy result.
        
    '''		
    model1 = small()
    model2 = medium()
    model3 = Large()
    bs = 128
    
    res = []
    for i in range(cycle):
        #load_model_weight(model1,model2,small_to_big=False)
        optim = torch.optim.Adam(model1.parameters(), lr=0.001)
        res_ = train(model1,loss,optim,down_sampled_train_small,label,epoch1,bs,device)
        res += res_
        
        model2 = medium()
        load_model_weight(model1,model2,small_to_big=True,first = True)
        optim = torch.optim.Adam(model2.parameters(), lr=0.0005)
        res_ = train(model2,loss,optim,concat_train_medium,label,epoch2,bs,device)
        res += res_
        
        model3 = Large()
        load_model_weight(model2,model3,small_to_big=True,first = True)
        optim = torch.optim.Adam(model3.parameters(), lr=0.0002)
        res_ = train(model3,loss,optim,concat_train_large,label,epoch3,bs,device)
        res += res_

    return res

ep1 = 500
ep2 = 300
ep3 = 200
result = [0]*(ep1+ ep2+ep3)
result = np.array(result).astype('float64')

from timeit import default_timer as timer

start = timer()
for i in range(10):
    print(f'Run number {i}')
    out = cycle_train(ep1,ep2,ep3,cycle=1,loss=loss,res =result) 
    out = np.array(out)
    result += out
end = timer()
time_diff = end - start

result = result/10
small_plot = result[0:ep1]
position  = np.arange(0,ep1)
plt.plot(position,small_plot,'b')

medium_plot = result[ep1:ep1+ep2]
position  = np.arange(ep1,ep1+ep2)
plt.plot(position,medium_plot,'g')

large_plot = result[ep1+ep2:ep1+ep2+ep3]
position  = np.arange(ep1+ep2,ep1+ep2+ep3)
plt.plot(position,large_plot,'r')

plt.ylabel('loss')
plt.xlabel('iterations')
plt.plot(0,0,'b',label='small network')
plt.plot(0,0,'g', label = 'medium network')
plt.plot(0,0,'r',label='large network')
plt.plot(0,0,'y',label = 'time:' + str(time_diff/10))
plt.plot(0,0,'y',label = f' max reached : {np.max(result)}')
plt.legend(loc='best')
plt.show()