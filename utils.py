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

		model2_l1_weight = model2.l1.weight.clone().detach()
		model2_l2_weight = model2.l2.weight.clone().detach()
		#print(model1.l1.weight.grad.cpu().detach().numpy())

		model2_l1_weight[128:,784:] = model1_l1_weight		
		model2_l2_weight[:,128:] = model1_l2_weight
		
		model2.l1.weight = torch.nn.Parameter(model2_l1_weight)
		model2.l2.weight = torch.nn.Parameter(model2_l2_weight)
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

