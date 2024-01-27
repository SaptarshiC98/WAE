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

batch_size = 100
epochs = 500
lr = 0.0001
dim_h = 128
n_z = 8
LAMBDA = 10
n_channel = 3
sigma = 1

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.n_channel =n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, n_channel, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.main(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x


    
OUTPUT = torch.load('data_2.pt')    
num_n = 5
n_rep = 10
test_errors = np.zeros((num_n, n_rep))
batch_size = 100
epochs = 10
lr = 0.0001
dim_h = 128
n_z = 8
LAMBDA = 10
n_channel = 3
sigma = 1

for rep in range(n_rep):
    for n_iter in range(num_n):
        encoder, decoder, discriminator = Encoder(), Decoder(), Discriminator()
        criterion = nn.MSELoss()

        encoder.train()
        decoder.train()
        discriminator.train()

        # Optimizers
        enc_optim = optim.Adam(encoder.parameters(), lr=lr)
        dec_optim = optim.Adam(decoder.parameters(), lr=lr)
        dis_optim = optim.Adam(discriminator.parameters(), lr=0.5 * lr)

        enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
        dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)

        n = (n_iter + 1) * 200
        train_loader = DataLoader(dataset=OUTPUT[0: n, :, :, :],
                                  batch_size=batch_size,
                                  shuffle=True)
        if torch.cuda.is_available():
            encoder, decoder, discriminator = encoder.cuda(), decoder.cuda(), discriminator.cuda()

        one = torch.Tensor([1])
        mone = one * -1

        if torch.cuda.is_available():
            one = one.cuda()
            mone = mone.cuda()

        # step = 0
        for epoch in range(epochs):
            for batch in tqdm(train_loader, disable=True):
                images = batch
                images = images.to(device)

                encoder.zero_grad()
                decoder.zero_grad()
                discriminator.zero_grad()

                frozen_params(decoder)
                frozen_params(encoder)
                free_params(discriminator)

                z_fake = torch.randn(images.size()[0], n_z) * sigma
                z_fake = z_fake.to(device)

                d_fake = discriminator(z_fake)

                z_real = encoder(images)
                d_real = discriminator(z_real)

                loss_1 = -torch.log(d_fake).mean() - torch.log(1 - d_real).mean()

                loss_1.backward()

                dis_optim.step()

                free_params(decoder)
                free_params(encoder)
                frozen_params(discriminator)

                batch_size = images.size()[0]

                z_real = encoder(images)
                x_recon = decoder(z_real)
                d_real = discriminator(encoder(Variable(images.data)))

                loss_2 = criterion(x_recon, images) - LAMBDA * (torch.log(d_real)).mean()
                loss_2.backward()

                enc_optim.step()
                dec_optim.step()

        x_real = OUTPUT[10000:11000, :, :, :]
        x_real = x_real.to(device)
        z_real = encoder(x_real)
        x_recon = decoder(z_real)
        simu = decoder(z_fake)
        test_errors[n_iter, rep] = criterion(x_recon, x_real)  

        
torch.save(test_errors, 'recon_2.pt')          