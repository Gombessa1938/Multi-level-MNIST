import torch
import torch.nn as nn
import torch.nn.functional as F

def load_model_weight(model1,model2,small_to_big = True):
	'''
	Load layer weight from model1 to model2
	'''
	if small_to_big == True:
		model1_l1_weight = model1.l1.weight.clone().detach()
		model1_l2_weight = model1.l2.weight.clone().detach()
		model1_l3_weight = model1.l3.weight.clone().detach()
		model1_l4_weight = model1.l4.weight.clone().detach()
		model1_l5_weight = model1.l5.weight.clone().detach()
		zero = torch.zeros(128,1024)
		concat_weight = torch.zeros(1024+256,128)
		concat_weight = torch.cat((zero,model1_l1_weight),dim=1)
		model2.l1.weight = torch.nn.Parameter(concat_weight)
		model2.l2.weight = torch.nn.Parameter(model1_l2_weight)
		model2.l3.weight = torch.nn.Parameter(model1_l3_weight)
		model2.l4.weight = torch.nn.Parameter(model1_l4_weight)
		model2.l5.weight = torch.nn.Parameter(model1_l5_weight)
	else:
		new_small_l1_weight = model1.l1.weight.clone().detach()
		new_l1_weight = new_small_l1_weight[:,1024:1024+256]
		model_new_small = model2
		model_new_small.l1.weight = torch.nn.Parameter(new_l1_weight)
		model_new_small.l2.weight = torch.nn.Parameter(model1.l2.weight.clone().detach())
		model_new_small.l3.weight = torch.nn.Parameter(model1.l3.weight.clone().detach())
		model_new_small.l4.weight = torch.nn.Parameter(model1.l4.weight.clone().detach())
		model_new_small.l5.weight = torch.nn.Parameter(model1.l5.weight.clone().detach())