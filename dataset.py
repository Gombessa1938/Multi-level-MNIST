import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image
from tqdm import tqdm
import numpy as np 

data = np.load('/Users/joe/Documents/llnl_branch/llnl.npz')
data = data['u']
print(data.shape)
#making downsampled image
down_sampled_train = torch.zeros(50000,16,16)
for i in tqdm(range(50000)):
  img = data[i].reshape(32,32)
  img_c = np.array(Image.fromarray(img).resize((16, 16), Image.LANCZOS))
  img_c = img_c.astype('float32')
  down_sampled_train[i] = torch.from_numpy(img_c)  

#making difference image
difference_train = torch.zeros(50000,32,32)
for i in tqdm(range(50000)):
  first = data[i].reshape(32,32)
  small = down_sampled_train[i].numpy()
  up_sample = np.array(Image.fromarray(small).resize((32, 32), Image.LANCZOS)) #upsample image
  first = first.astype('float32')
  up_sample = up_sample.astype('float32')
  difference_train[i] = torch.from_numpy(first - up_sample)

#concat difference image and coarse iamge 
concat_train = torch.zeros(50000,32*32 + 16*16)
for i in tqdm(range(50000)):
  concat_train[i] = torch.cat((difference_train[i].reshape(32*32,),\
    down_sampled_train[i].reshape(16*16,)))
  
  
  
#-----------------------------------------
down_sampled_train_small = torch.zeros(50000,8,8)
for i in tqdm(range(50000)):
  img = data[i].reshape(32,32)
  img_c = np.array(Image.fromarray(img).resize((8, 8), Image.LANCZOS))
  img_c = img_c.astype('float32')
  down_sampled_train_small[i] = torch.from_numpy(img_c) 
  
difference_train_medium = torch.zeros(50000,16,16)
for i in tqdm(range(50000)):
  first = data[i].reshape(32,32)
  small = down_sampled_train_small[i].numpy()
  up_sample = np.array(Image.fromarray(small).resize((16, 16), Image.LANCZOS)) #upsample image
  first = first.astype('float32')
  up_sample = up_sample.astype('float32')
  
concat_train_medium = torch.zeros(50000,16*16 + 8*8)
for i in tqdm(range(50000)):
  concat_train_medium[i] = torch.cat((difference_train_medium[i].reshape(16*16,),\
    down_sampled_train_small[i].reshape(8*8,))) 
  
  
concat_train_large = torch.zeros(50000,32*32+16*16+8*8)
for i in tqdm(range(50000)):
  concat_train_large[i] = torch.cat((difference_train[i].reshape(32*32,),concat_train_medium[i].reshape(16*16+8*8,))) 


np.save('concat_train_medium.npy',concat_train_medium)
np.save('down_sampled_train_small',down_sampled_train_small)
np.save('concat_train_large.npy',concat_train_large)