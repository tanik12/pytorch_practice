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

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

class ImageTransform():
    #img size is resized, normalizated and data argumented
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(
                        resize, scale=(0.5, 1.0)), #data argument
                    transforms.RandomHorizontalFlip(), #data argument
                    transforms.ToTensor(), # data to tensor
                    transforms.Normalize(mean, std) #normalize
                ]),
                'val': transforms.Compose([
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            }
    def __call__(self, img, phase='train'):
        # select train or val
        return self.data_transform[phase](img)

#read img
image_file_path = '/Users/gisen/git/pytorch_advanced/1_image_classification/data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)

#show img
plt.imshow(img)
plt.show()

#pre-preprocessing img and show preprocessing img
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = ImageTransform(size, mean, std)
img_transformed = transform(img, phase='train') #torch.Size([3, 224, 224])

#(chanel, hight, weight)⇨ (hight, weight, chanel). And transform to value range 0~1.
img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.show()
