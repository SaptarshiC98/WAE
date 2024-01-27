import nltk
nltk.download('wordnet')

import os
import json
import torch
import numpy as np
from pytorch_pretrained_biggan import (BigGAN,one_hot_from_names,truncated_noise_sample)

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import  pytorch_fid_wrapper as pfw

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

model = BigGAN.from_pretrained('biggan-deep-128')
model = model.to(device)
latent_dim = 2
num_samples = 11000
OUTPUT = torch.zeros(size=[num_samples,3,128,128])
batch_size = 100
class_name = 'soap bubble'
class_id =0
truncation = 1
NUM_IMAGENET_CLS = 1000
start_batch_num = 0
num_batches = num_samples // batch_size
for b in range(start_batch_num, num_batches):
    print('Batch: {}/{}'.format(b+1, num_batches))

    # Prepare inputs
    if class_id:
      class_vector = np.zeros((args.batch_size, NUM_IMAGENET_CLS), dtype=np.float32)
      class_vector[:, class_id] = 1
    elif class_name:
      class_vector = one_hot_from_names([class_name], batch_size = batch_size)
    else:
      raise Exception("Must specify either class name or ID!")
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size= batch_size)
    latent_dim_orig = noise_vector.shape[1]
    if latent_dim_orig != latent_dim:
      #Reduce dimension of noise_vector by fixing components to be zero
      assert latent_dim < latent_dim_orig
      k = latent_dim_orig - latent_dim
      noise_vector[:,:k] = 0

    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)
    noise_vector = noise_vector.to(device)
    class_vector = class_vector.to(device)
    with torch.no_grad():
      output = model(noise_vector, class_vector, truncation)
    output = (output + 1)*(0.5)
    OUTPUT[b* batch_size: batch_size*(b+1),:,:,:] = output

    
trans = transforms.Compose([transforms.Resize(28)])
OUTPUT = trans(OUTPUT)

torch.save(OUTPUT, 'data_2.pt')    