from pyexpat import model
import numpy as np 
import torch
from matplotlib import pyplot as plt

def train(input_model,loss,optimizer,datasets,label,epoch,batch_size,device,losses):
    model = input_model
    model = model.to(device)
    loss_function = loss
    optim = optimizer
    batch_size = batch_size
    losses= []
    data = datasets
    data = data.to(device)
    if data.shape[1] == 16:
        flat_size = data.shape[1]*data.shape[1]
    elif data.shape[1] == 8:
        flat_size = data.shape[1]*data.shape[1]
    else:
        flat_size = data.shape[1]
    for i in range(epoch):
        samp = np.random.randint(0, data.shape[0], size=(batch_size)) 
        X = data[samp].reshape((-1, flat_size))
        Y = label[samp]
        model.zero_grad()
        output = model(X)
        loss = loss_function(output,Y)
        loss.backward()
        optim.step()
        losses.append(loss.detach().clone().numpy())
    return losses