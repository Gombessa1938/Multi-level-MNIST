from pyexpat import model
import numpy as np 
import torch
from matplotlib import pyplot as plt
torch.manual_seed(42)
def train(input_model,loss,optimizer,datasets,label,epoch,batch_size):
    model = input_model
    loss_function = loss
    optim = optimizer
    batch_size = batch_size
    accuracies= []
    data = datasets

    if data.shape[1] == 14:
        flat_size = data.shape[1]*data.shape[1]
    else:
        flat_size = data.shape[1]
    
    for i in range(epoch):
        samp = np.random.randint(0, data.shape[0], size=(batch_size)) 
        X = data[samp].reshape((-1, flat_size))
        Y = label[samp]
        model.zero_grad()
        output = model(X)
        cat = torch.argmax(output,dim=1)
        accuracy=(cat==Y).float().mean()
        loss = loss_function(output,Y)
        loss = loss.mean()
        loss.backward()
        optim.step()
        accuracy = accuracy.item()
        accuracies.append(accuracy)
    return accuracies