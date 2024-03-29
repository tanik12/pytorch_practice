import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

class Generator(nn.Module):
    
    def __init__(self, z_dim=20, image_size=64):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(z_dim, image_size * 8,
                                   kernel_size=4, stride=1),
                nn.BatchNorm2d(image_size * 8),
                nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 8, image_size * 4,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size * 4),
                nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 4, image_size * 2,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size * 2),
                nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 2, image_size,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size),
                nn.ReLU(inplace=True))

        self.last = nn.Sequential(
                nn.ConvTranspose2d(image_size, 1,
                                   kernel_size=4, stride=2, padding=1),
                nn.Tanh())
        # 注意：白黒画像なので出力チャネルは１つだけ
        
    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out

class Discriminator(nn.Module):

    def __init__(self, z_dim=20, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
                nn.Conv2d(1, image_size, kernel_size=4,
                          stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True))
        #注意：白黒画像なので入力チャネルは１つだけ

        self.layer2 = nn.Sequential(
                nn.Conv2d(image_size, image_size*2, kernel_size=4,
                          stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
                nn.Conv2d(image_size*2, image_size*4, kernel_size=4,
                          stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(
                nn.Conv2d(image_size*4, image_size*8, kernel_size=4,
                          stride=2, padding=1),
                    nn.LeakyReLU(0.1, inplace=True))

        self.last = nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out
