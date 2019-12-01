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

def make_datapath_list(phase='train'):
    rootpath = "/Users/gisen/git/pytorch_advanced/1_image_classification/data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    #print(target_path)

    path_list = []

    #globを用いることで検索に指定した文字にマッチしたwardを出力してくれる。.
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

if __name__ == '__main__':
    train_list = make_datapath_list(phase='train')
    val_list = make_datapath_list(phase='val')

    #print(train_list)
