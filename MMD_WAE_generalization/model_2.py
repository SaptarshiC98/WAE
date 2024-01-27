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
epochs = 100
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

def rbf_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x)
        res1 += torch.exp(-C * dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats

def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats

    
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
        encoder, decoder = Encoder(), Decoder()
        criterion = nn.MSELoss()

        encoder.train()
        decoder.train()

        # Optimizers
        enc_optim = optim.Adam(encoder.parameters(), lr=lr)
        dec_optim = optim.Adam(decoder.parameters(), lr=lr)

        enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)

        n = (n_iter + 1) * 1000
        train_loader = DataLoader(dataset=OUTPUT[0: n, :, :, :],
                                  batch_size=batch_size,
                                  shuffle=True)
        if torch.cuda.is_available():
            encoder, decoder = encoder.cuda(), decoder.cuda()

        one = torch.Tensor([1])
        mone = one * -1

        if torch.cuda.is_available():
            one = one.cuda()
            mone = mone.cuda()

        # step = 0
        for epoch in range(epochs):
            step = 0
            for images in tqdm(train_loader, disable=True):
                if torch.cuda.is_available():
                    images = images.cuda()

                enc_optim.zero_grad()
                dec_optim.zero_grad()

        # ======== Train Generator ======== #

                batch_size = images.size()[0]

                z = encoder(images)
                x_recon = decoder(z)

                recon_loss = criterion(x_recon, images)

        # ======== MMD Kernel Loss ======== #

                z_fake = Variable(torch.randn(images.size()[0], n_z) * sigma)
                if torch.cuda.is_available():
                    z_fake = z_fake.cuda()
                z_real = encoder(images)
                mmd_loss = imq_kernel(z_real, z_fake, h_dim=n_z)
                mmd_loss = mmd_loss / batch_size

                total_loss = recon_loss + 10* mmd_loss
                total_loss.backward()

                enc_optim.step()
                dec_optim.step()

                step += 1

        z_fake = torch.randn(100, n_z) * sigma
        z_fake = z_fake.to(device)
        simu = decoder(z_fake)
        test_errors[n_iter, rep] = pfw.fid(simu, OUTPUT[10000:11000, :, :, :],
                                           device=device)
       # print("n_iter = %d, repeat = %d, Test Error = %.4f" % (n_iter + 1, rep + 1, test_errors[n_iter, rep]))    

        
torch.save(test_errors, 'test_errors_2.pt')          