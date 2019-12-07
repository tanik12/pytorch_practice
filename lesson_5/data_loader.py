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

rootpath = "/Users/gisen/git/pytorch_advanced/5_gan_generation/data/"

def make_datapath_list():
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。"""

    train_img_list = list() #画像ファイルパスを格納

    for img_idx in range(200):
        img_path = rootpath + "img_78/img_7_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

        img_path = rootpath + "img_78/img_8_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

    return train_img_list

class ImageTransform():
    #画像の前処理クラス

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    def __call__(self, img):
        return self.data_transform(img)

class GAN_img_Dataset(data.Dataset):
    #画像のDatasetクラス。PyoTorchのDatasetクラスを継承
    
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        #画像の枚数を返す
        return len(self.file_list)

    def __getitem__(self, index):
        #前処理した画像のTensor形式のデータを取得
        img_path = self.file_list[index]
        img = Image.open(img_path) #[高さ][幅]白黒

        #画像の前処理
        img_transformed = self.transform(img)

        return img_transformed

if __name__ == "__main__":
    #DataLoaderの作成と動作確認
    #ファイルリストを作成
    train_img_list = make_datapath_list()

    #Datasetを作成
    mean = (0.5,)
    std = (0.5,)
    train_dataset = GAN_img_Dataset(
            file_list=train_img_list, transform=ImageTransform(mean, std))

    #DataLoaderを作成
    batch_size = 64

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

    #動作確認
    batch_iterator = iter(train_dataloader) #イテレータに変換
    images = next(batch_iterator) #１番目の要素を取り出す
    print(images.size())
