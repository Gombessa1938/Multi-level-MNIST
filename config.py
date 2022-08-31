import torch.nn as nn 
import torch 
import numpy as np 
import torchvision.datasets as datasets
import torchvision.transforms as tf

device = "cpu"
image_full = 64
image_coarse = 32
model_layer_level=5
batch_size = 64
loss = nn.NLLLoss()
diff_dim_x = 32
diff_dim_y = 32

down_sampled_train_small = torch.from_numpy(np.load('/Users/joe/Documents/mnist_branch/mnist_down_sampled_train_small_64_ds_diff_no_diff.npy'))
concat_train_medium = torch.from_numpy(np.load('/Users/joe/Documents/mnist_branch/mnist_concat_train_medium_64_ds_diff_no_diff.npy'))
concat_train_large = torch.from_numpy(np.load('/Users/joe/Documents/mnist_branch/mnist_concat_train_large_64_ds_diff_no_diff.npy'))
mnist_trainset = datasets.MNIST(root='/Users/joe/Documents/mnist_data/data', train=True, download=True, transform=tf.ToTensor())