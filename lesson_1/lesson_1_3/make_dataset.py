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

def make_datapath_list(phase='train'):
    rootpath = "/Users/gisen/git/pytorch_advanced/1_image_classification/data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    #print(target_path)

    path_list = []

    #globを用いることで検索に指定した文字にマッチしたwardを出力してくれる。.
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list, rootpath

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

class HymenopteraDataset(data.Dataset):
    #アリと鉢の画像のDataset class. PyTorchのDataset classを継承。
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list #filepath list
        self.transform = transform #instance of preprocessing class
        self.phase = phase # train or val

    def __len__(self):
        #return sum of image number
        return len(self.file_list)

    def __getitem__(self, index):
        #前処理した画像のTensor形式のデータとラベルを取得
        
        #load img(index number)
        img_path = self.file_list[index]
        img = Image.open(img_path) #[height][width][chanel]

        #preprocessing
        img_transformed = self.transform(
                img, self.phase) #torch.Size([3, 224, 224])

        #画像のラベルをファイル名から抜き出す
        #id_num = len(rootpath+phase)
        if self.phase == "train":
            str_num_train = str_num + len("train/")
            label = img_path[str_num_train:str_num_train+4]
        elif self.phase == "val":
            str_num_val = str_num + len("val/")
            label = img_path[str_num_val:str_num_val+4]

        #ラベルを数値に変更する
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label

if __name__ == "__main__":
    train_list, rootpath_train = make_datapath_list(phase='train')
    val_list, _ = make_datapath_list(phase='val')
    #print(len(rootpath_train), rootpath_train)
    str_num = len(rootpath_train)

    #pre-preprocessing img and show preprocessing img
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    ####

    train_dataset = HymenopteraDataset(
            file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = HymenopteraDataset(
            file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

    #動作確認
    index = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

