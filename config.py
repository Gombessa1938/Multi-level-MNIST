import torch.nn as nn 
device = "cpu"
image_full = 32
image_coarse = 16
model_layer_level=5
data_path = " "
cycle = 2
loss = nn.MSELoss()