import torch.nn as nn 
device = "cpu"
image_full = 64
image_coarse = 32
model_layer_level=5
data_path = " "
cycle = 2
loss = nn.NLLLoss()#nn.CrossEntropyLoss()#nn.MSELoss()