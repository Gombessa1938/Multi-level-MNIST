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
		zero = torch.zeros(128,784)
		concat_weight_second_half = torch.zeros(128,980)
		concat_weight_first_half = torch.cat((zero,model1_l1_weight),dim=1)
		concat = torch.cat((concat_weight_second_half,concat_weight_first_half),dim=0)
		l2_zero = torch.zeros(20,128)
		concat_l2 = torch.cat((l2_zero,model1_l2_weight),dim=1)
		model2.l1.weight = torch.nn.Parameter(concat)
		model2.l2.weight = torch.nn.Parameter(concat_l2)
		model2.l3.weight = torch.nn.Parameter(model1_l3_weight)
	else:
		l1 = model1.l1.weight.clone().detach()
		l2 = model1.l2.weight.clone().detach()
		new_l1_weight = l1[128:,784:980]
		new_l2_weight =  l2[:,128:]

		model_new_small = model2		
		model_new_small.l1.weight = torch.nn.Parameter(new_l1_weight)
		model_new_small.l2.weight = torch.nn.Parameter(new_l2_weight)
		model_new_small.l3.weight = torch.nn.Parameter(model1.l3.weight.clone().detach())
