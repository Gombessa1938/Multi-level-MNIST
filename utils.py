import torch
import torch.nn as nn
import torch.nn.functional as F

def load_model_weight(model1,model2,small_to_big = True,first = False):
	'''
	Load layer weight from model1 to model2
	'''
	if small_to_big == True:
     
		#adding each model layer weight into lists
		model1_weight_list = []
		for name, param in model1.named_parameters():
			model1_weight_list.append(param.clone().detach())
		model2_weight_list = []
		for name,param in model2.named_parameters():
			model2_weight_list.append(param.clone().detach())
   
		loaded_weight = []
		#loading model1 weight into model2 weight block
		for model1_weight, model2_weight in zip(model1_weight_list,model2_weight_list):
			x,y = model1_weight.shape[0],model1_weight.shape[1]
			xx,yy = model2_weight.shape[0],model2_weight.shape[1]
			if first:
				model2_weight = torch.zeros_like(model2_weight)
			model2_weight[xx-x:,yy-y:] = model1_weight
			loaded_weight.append(model2_weight)

		model2.l1.weight = torch.nn.Parameter(loaded_weight[0])
		model2.l2.weight = torch.nn.Parameter(loaded_weight[1])
		model2.l3.weight = torch.nn.Parameter(loaded_weight[2])
		model2.l4.weight = torch.nn.Parameter(loaded_weight[3])
		model2.l5.weight = torch.nn.Parameter(loaded_weight[4])

	else:
		new_small_l1_weight = model1.l1.weight.clone().detach()
		new_l1_weight = new_small_l1_weight[:,1024:1024+256]
		model_new_small = model2
		model_new_small.l1.weight = torch.nn.Parameter(new_l1_weight)
		model_new_small.l2.weight = torch.nn.Parameter(model1.l2.weight.clone().detach())
		model_new_small.l3.weight = torch.nn.Parameter(model1.l3.weight.clone().detach())
		model_new_small.l4.weight = torch.nn.Parameter(model1.l4.weight.clone().detach())
		model_new_small.l5.weight = torch.nn.Parameter(model1.l5.weight.clone().detach())