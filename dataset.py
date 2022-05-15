import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image
from tqdm import tqdm
import numpy as np 

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

#making downsampled image
down_sampled_train = torch.zeros(60000,14,14)
for i in tqdm(range(60000)):
  img = mnist_trainset.data[i].numpy()
  img_c = np.array(Image.fromarray(img).resize((14, 14), Image.LANCZOS))
  img_c = img_c.astype('float32')
  down_sampled_train[i] = torch.from_numpy(img_c)  

#making difference image
difference_train = torch.zeros(60000,28,28)
for i in tqdm(range(60000)):
  first = mnist_trainset.data[i].numpy()
  small = down_sampled_train[i].numpy()
  up_sample = np.array(Image.fromarray(small).resize((28, 28), Image.LANCZOS)) #upsample image
  first = first.astype('float32')
  up_sample = up_sample.astype('float32')
  difference_train[i] = torch.from_numpy(first - up_sample)

#concat difference image and coarse iamge 
concat_train = torch.zeros(60000,28*28 + 14*14)
for i in tqdm(range(60000)):
  concat_train[i] = torch.cat((difference_train[i].reshape(28*28,),down_sampled_train[i].reshape(14*14,)))

np.save('concat_train.npy',concat_train)
np.save('down_sampled_train',down_sampled_train)