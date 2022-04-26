import os
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfm
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class _G(nn.Module):
    def __init__(self, nz, nc, ngf):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 8, 1, 0), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(ngf * 2, nc, 3, 1, 1), nn.Tanh())

    def forward(self, z):
        return self.gen(z)


class _E(nn.Module):
    def __init__(self, nz, ndf):
        super().__init__()
        self.ebm = nn.Sequential(nn.Linear(nz, ndf), nn.LeakyReLU(0.2),
                                 nn.Linear(ndf, ndf), nn.LeakyReLU(0.2),
                                 nn.Linear(ndf, 1))

    def forward(self, z):
        return self.ebm(z.squeeze()).view(-1, 1, 1, 1)