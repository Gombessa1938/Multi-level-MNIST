import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image
from tqdm import tqdm
import numpy as np 
from matplotlib import pyplot as plt

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

mnist_upsample = torch.zeros(60000,32,32)
for i in tqdm(range(60000)):
  img = mnist_trainset.data[i].numpy()
  img_up = np.array(Image.fromarray(img).resize((32,32),Image.LANCZOS)) #Image.BICUBIC
  img_up = img_up.astype('float32')
  mnist_upsample[i] = torch.from_numpy(img_up)

#data = np.load('/Users/joe/Documents/llnl_branch/llnl.npz')
data = mnist_upsample.numpy()
#making downsampled image
down_sampled_train = torch.zeros(60000,16,16)
for i in tqdm(range(60000)):
  img = data[i].reshape(32,32)
  img_c_ = np.array(Image.fromarray(img).resize((16, 16), Image.BICUBIC))
  img_c_ = img_c_.astype('float32')
  down_sampled_train[i] = torch.from_numpy(img_c_)  

# img_small = down_sampled_train[0]
# plt.imshow(img_small)
# plt.show()
#making difference image
difference_train = torch.zeros(60000,32,32)
for i in tqdm(range(60000)):
  first = data[i].reshape(32,32)
  small = down_sampled_train[i].numpy()
  up_sample = np.array(Image.fromarray(small).resize((32, 32), Image.BICUBIC)) #upsample image
  first = first.astype('float32')
  up_sample = up_sample.astype('float32')
  diff = first #- up_sample
  difference_train[i]  = torch.from_numpy(first)#torch.from_numpy(np.array(Image.fromarray(diff).resize((32,32),Image.BICUBIC)))
  
#concat difference image and coarse iamge 
concat_train = torch.zeros(60000,32*32 + 16*16)
for i in tqdm(range(60000)):
  concat_train[i] = torch.cat((difference_train[i].reshape(32*32,),\
    down_sampled_train[i].reshape(16*16,)))
  
  
#-----------------------------------------
down_sampled_train_small = torch.zeros(60000,8,8)
for i in tqdm(range(60000)):
  img = data[i].reshape(32,32)
  img_c = np.array(Image.fromarray(img).resize((8, 8), Image.LANCZOS))
  img_c = img_c.astype('float32')
  down_sampled_train_small[i] = torch.from_numpy(img_c) 
  
# img_small = down_sampled_train_small[0]
# plt.imshow(img_small)
# plt.show()
  
difference_train_medium = torch.zeros(60000,16,16)
for i in tqdm(range(60000)):
  #first = data[i].reshape(64,64)
  small = down_sampled_train_small[i].numpy()
  up_sample = np.array(Image.fromarray(small).resize((16, 16), Image.LANCZOS)) #upsample image
  first = img_c_#first.astype('float32')
  up_sample = up_sample.astype('float32')
  diff = down_sampled_train[i] #- up_sample
  difference_train_medium[i]  = diff#torch.from_numpy(np.array(Image.fromarray(diff).resize((32,32), Image.BICUBIC)))

     
concat_train_medium = torch.zeros(60000,16*16 + 8*8)
for i in tqdm(range(60000)):
  concat_train_medium[i] = torch.cat((difference_train_medium[i].reshape(16*16,),\
    down_sampled_train_small[i].reshape(8*8,))) 
 
concat_train_large = torch.zeros(60000,32*32+16*16+8*8)
for i in tqdm(range(60000)):
  concat_train_large[i] = torch.cat((difference_train[i].reshape(32*32,),concat_train_medium[i].reshape(16*16+8*8,))) 


np.save('mnist_concat_train_medium_64_ds_diff_no_diff.npy',concat_train_medium)
np.save('mnist_down_sampled_train_small_64_ds_diff_no_diff.npy',down_sampled_train_small)
np.save('mnist_concat_train_large_64_ds_diff_no_diff.npy',concat_train_large)


