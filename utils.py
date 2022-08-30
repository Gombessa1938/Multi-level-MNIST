import torch
import torch.nn as nn
torch.manual_seed(42)

def load_model_weight(model1,model2,small_to_big = True,first = False):
    '''
    This function load layer weight from model1 to model2 automatically
    Args:
        model1: Pytorch model
        model2: Pytorch model
        small_to_big: Bool
        if True then model1 is the smaller model, else then model 2 is the 
        smaller model.
        first: Bool
        if first is True then rest network weight will be set to 0, else it keeps the 
        network weight that are not being effected.
    '''
    #adding each model layer weight into lists
    model1_weight_list = []
    for _, param in model1.named_parameters():
        model1_weight_list.append(param.clone().detach())
    model2_weight_list = []
    for _,param in model2.named_parameters():
        model2_weight_list.append(param.clone().detach())

    #loading model1 weight into model2 weight block
    counter = 0
    for model1_weight, model2_weight in zip(model1_weight_list,model2_weight_list):
        x,y = model1_weight.shape[0],model1_weight.shape[1]
        xx,yy = model2_weight.shape[0],model2_weight.shape[1]
        if first:
            model2_weight = torch.zeros_like(model2_weight)
        if small_to_big==True:
            model2_weight[xx-x:,yy-y:] = model1_weight #load weight
        else:
            model2_weight = model1_weight[x-xx:,y-yy:]
        model2_weight_list[counter] = model2_weight
        counter +=1

    #load weight to model
    for i in range(len(model2.linears)):
        model2.linears[i].weight = nn.Parameter(model2_weight_list[i])

def make_layer_input_list(config):
    '''
    returns two list each contain input number for building network
    Example:
        level=5:
        large = [256,128,64,16]
        medium = [128,64,32,16]
        small = [64,32,16,16]
    '''
    level = config.model_layer_level
    image_coarse = config.image_coarse

    layer_large = []
    input_size = image_coarse*image_coarse
    for i in range(level-1):
        layer_large.append(input_size)
        if i == level -3:
            input_size = input_size //4
        else:
            input_size = input_size//2

    layer_medium= [0]*len(layer_large)
    for i in range(len(layer_large)):
        if i == len(layer_large)-1:
            layer_medium[i] = layer_large[i]
        else:
            layer_medium[i] = layer_large[i]//2
   
    layer_small = [0]*len(layer_medium)
    for i in range(len(layer_small)):
        if i ==len(layer_large)-1:
            layer_small[i] = layer_large[i]
        else:
            layer_small[i] = layer_medium[i]//2
   
    return layer_large,layer_medium,layer_small