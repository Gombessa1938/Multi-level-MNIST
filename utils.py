import torch
import torch.nn as nn
torch.manual_seed(42)
def load_model_weight(model1,model2,small_to_big = True,first = False):
	'''
	Load layer weight from model1 to model2
	'''
	# if small_to_big == True:
	#adding each model layer weight into lists
	model1_weight_list = []
	for name, param in model1.named_parameters():
		model1_weight_list.append(param.clone().detach())
	model2_weight_list = []
	for name,param in model2.named_parameters():
		model2_weight_list.append(param.clone().detach())

	#loading model1 weight into model2 weight block
	counter = 0
	for model1_weight, model2_weight in zip(model1_weight_list,model2_weight_list):
		x,y = model1_weight.shape[0],model1_weight.shape[1]
		xx,yy = model2_weight.shape[0],model2_weight.shape[1]
		#if first:
		model2_weight = torch.zeros_like(model2_weight)
		if small_to_big==True:
			model2_weight[xx-x:,yy-y:] = model1_weight #load weight
		else:
			model2_weight = model1_weight[x-xx:,y-yy:]
		model2_weight_list[counter] = model2_weight
		counter +=1

	#load weight to model
	for i in range(len(model2.linears)):
		model2.linears[i].weight = torch.nn.Parameter(model2_weight_list[i])
