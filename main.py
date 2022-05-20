import torch
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torch.nn as nn
import numpy as np
from models import small,Large
from train import train 
from utils import load_model_weight

#data loading
data = np.load('/Users/joe/Documents/llnl.npz')
label = torch.from_numpy(data['Q'].astype('float32'))
down_sampled_train = torch.from_numpy(np.load('down_sampled_train.npy'))
concat_train = torch.from_numpy(np.load('concat_train.npy'))

def cycle_train(epoch1,epoch2,epoch3,counter):
	'''
	complete one training cycle
	small network -> large network -> small network 
	'''		
	model1 = small()
	model2 = Large()
	#load_model_weight(model1,model2,small_to_big=False)
	loss = nn.MSELoss()
	optim = torch.optim.Adam(model1.parameters(), lr=0.0001)
	train(model1,loss,optim,down_sampled_train,label,epoch1,128)

	model2 = Large()
	load_model_weight(model1,model2,small_to_big=True)
	optim = torch.optim.Adam(model2.parameters(), lr=0.001)
	train(model2,loss,optim,concat_train,label,epoch2,128)

	model1 = small()
	load_model_weight(model2,model1,small_to_big=False)
	optim = torch.optim.Adam(model1.parameters(), lr=0.0001)
	train(model1,loss,optim,down_sampled_train,label,epoch3,128)
 
#===============================

	load_model_weight(model1,model2,small_to_big=False)
	optim = torch.optim.Adam(model1.parameters(), lr=0.0001)
	train(model1,loss,optim,down_sampled_train,label,10,128)

	model2 = Large()
	load_model_weight(model1,model2,small_to_big=True)
	optim = torch.optim.Adam(model2.parameters(), lr=0.001)
	train(model2,loss,optim,concat_train,label,10,128)

	model1 = small()
	load_model_weight(model2,model1,small_to_big=False)
	optim = torch.optim.Adam(model1.parameters(), lr=0.0001)
	train(model1,loss,optim,down_sampled_train,label,10,128)

cycle_train(50,50,50,counter = 1)
