import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image
from tqdm import tqdm
import numpy as np 
from matplotlib import pyplot as plt

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

mnist_upsample = torch.zeros(60000,64,64)
for i in tqdm(range(60000)):
  img = mnist_trainset.data[i].numpy()
  img_up = np.array(Image.fromarray(img).resize((64,64),Image.LANCZOS)) #Image.BICUBIC
  img_up = img_up.astype('float32')
  mnist_upsample[i] = torch.from_numpy(img_up)

#data = np.load('/Users/joe/Documents/llnl_branch/llnl.npz')
data = mnist_upsample.numpy()
print(data.shape)
#making downsampled image
down_sampled_train = torch.zeros(60000,32,32)
for i in tqdm(range(60000)):
  img = data[i].reshape(64,64)
  img_c_ = np.array(Image.fromarray(img).resize((32, 32), Image.BICUBIC))
  img_c_ = img_c_.astype('float32')
  down_sampled_train[i] = torch.from_numpy(img_c_)  

# img_small = down_sampled_train[0]
# plt.imshow(img_small)
# plt.show()
#making difference image
difference_train = torch.zeros(60000,64,64)
for i in tqdm(range(60000)):
  first = data[i].reshape(64,64)
  small = down_sampled_train[i].numpy()
  up_sample = np.array(Image.fromarray(small).resize((64, 64), Image.BICUBIC)) #upsample image
  first = first.astype('float32')
  up_sample = up_sample.astype('float32')
  difference_train[i] = torch.from_numpy(first - up_sample)

#concat difference image and coarse iamge 
concat_train = torch.zeros(60000,64*64 + 32*32)
for i in tqdm(range(60000)):
  concat_train[i] = torch.cat((difference_train[i].reshape(64*64,),\
    down_sampled_train[i].reshape(32*32,)))
  
  
  
#-----------------------------------------
down_sampled_train_small = torch.zeros(60000,16,16)
for i in tqdm(range(60000)):
  img = data[i].reshape(64,64)
  img_c = np.array(Image.fromarray(img).resize((16, 16), Image.LANCZOS))
  img_c = img_c.astype('float32')
  down_sampled_train_small[i] = torch.from_numpy(img_c) 
  

  
difference_train_medium = torch.zeros(60000,32,32)
for i in tqdm(range(60000)):
  first = data[i].reshape(64,64)
  small = down_sampled_train_small[i].numpy()
  up_sample = np.array(Image.fromarray(small).resize((32, 32), Image.LANCZOS)) #upsample image
  first = img_c_#first.astype('float32')
  up_sample = up_sample.astype('float32')
  difference_train_medium[i] = torch.from_numpy(first - up_sample)

# img_small = difference_train[0]
# plt.imshow(img_small)
# plt.show()
  
concat_train_medium = torch.zeros(60000,32*32 + 16*16)
for i in tqdm(range(60000)):
  concat_train_medium[i] = torch.cat((difference_train_medium[i].reshape(32*32,),\
    down_sampled_train_small[i].reshape(16*16,))) 
  
  
concat_train_large = torch.zeros(60000,64*64+32*32+16*16)
for i in tqdm(range(60000)):
  concat_train_large[i] = torch.cat((difference_train[i].reshape(64*64,),concat_train_medium[i].reshape(32*32+16*16,))) 


np.save('mnist_concat_train_medium_64.npy',concat_train_medium)
np.save('mnist_down_sampled_train_small_64',down_sampled_train_small)
np.save('mnist_concat_train_large_64.npy',concat_train_large)


