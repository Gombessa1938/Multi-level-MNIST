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

# k = 1100
# plt.imshow(concat_train_large[k][32*32+16*16:32*32+16*16+8*8].reshape(8,8).numpy())
# plt.show()
# plt.imshow(concat_train_large[k][32*32:32*32+16*16].reshape(16,16).numpy())
# plt.show()
# plt.imshow(concat_train_large[k][0:32*32].reshape(32,32).numpy())
# plt.show()


def cycle_train(epoch1,epoch2,epoch3,cycle,loss,res):
    '''
    complete one training cycle
    small network -> large network -> small network 
    '''		
    model1 = small()
    print(model1)
    model2 = medium()
    print(model2)
    model3 = Large()
    print(model3)
    bs = 128
    position = []
    l,r = 0,35
    
    res = []
    for i in range(cycle):
        #load_model_weight(model1,model2,small_to_big=False)
        optim = torch.optim.Adam(model1.parameters(), lr=0.001)
        res_ = train(model1,loss,optim,down_sampled_train_small,label,epoch1,bs,device,res)
        res += res_
        # position  = np.arange(l,r)
        # plt.plot(position,res,'b')
        # l  = r
        # r +=10
        
    
        model2 = medium()
        load_model_weight(model1,model2,small_to_big=True,first = True)
        optim = torch.optim.Adam(model2.parameters(), lr=0.0005)
        res_ = train(model2,loss,optim,concat_train_medium,label,epoch2,bs,device,res)
        res += res_
        # position = np.arange(l,r)
        # plt.plot(position,res,'g')
        # l = r
        # r +=10
        
        model3 = Large()
        load_model_weight(model2,model3,small_to_big=True,first = True)
        optim = torch.optim.Adam(model3.parameters(), lr=0.00005)
        res_ = train(model3,loss,optim,concat_train_large,label,epoch3,bs,device,res)
        res += res_
        # position = np.arange(l,r)
        # plt.plot(position,res,'r')

    return res

ep1 = 150
ep2 = 70
ep3 = 30
result = [0]*(ep1+ ep2+ep3)
result = np.array(result).astype('float64')

from timeit import default_timer as timer

start = timer()
for i in range(10):
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