import torch
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torch.nn as nn
import numpy as np
from models import small,Large
from train import train 
from utils import load_model_weight

#data loading
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=tf.ToTensor())
down_sampled_train = torch.from_numpy(np.load('down_sampled_train.npy'))
concat_train = torch.from_numpy(np.load('concat_train.npy'))

model1 = small()
loss = nn.NLLLoss(reduction='none')
optim = torch.optim.Adam(model1.parameters(), lr=0.0001)
train(model1,loss,optim,down_sampled_train,mnist_trainset.targets,300,128)

model2 = Large()
load_model_weight(model1,model2,small_to_big=True)
optim = torch.optim.Adam(model2.parameters(), lr=0.001)
train(model2,loss,optim,concat_train,mnist_trainset.targets,100,128)

model1 = small()
load_model_weight(model2,model1,small_to_big=False)
loss = nn.NLLLoss(reduction='none')
optim = torch.optim.Adam(model1.parameters(), lr=0.0001)
train(model1,loss,optim,down_sampled_train,mnist_trainset.targets,300,128)