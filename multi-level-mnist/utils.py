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
		zero = torch.zeros(128,784)
		concat_weight = torch.zeros(980,128)
		concat_weight = torch.cat((zero,model1_l1_weight),dim=1)
		model2.l1.weight = torch.nn.Parameter(concat_weight)
		model2.l2.weight = torch.nn.Parameter(model1_l2_weight)
	else:
		new_small_l1_weight = model1.l1.weight.clone().detach()
		new_l1_weight = new_small_l1_weight[:,784:980]
		model_new_small = model2
		model_new_small.l1.weight = torch.nn.Parameter(new_l1_weight)
		model_new_small.l2.weight = torch.nn.Parameter(model1.l2.weight.clone().detach())