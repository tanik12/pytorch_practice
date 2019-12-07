import matplotlib.pyplot as plt
from gan import Generator, Discriminator
from data_loader import make_datapath_list, ImageTransform, GAN_img_Dataset

from train import weights_init, train_model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import time

###########################
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
###########################

#動作確認
batch_iterator = iter(train_dataloader) #イテレータに変換
images = next(batch_iterator) #１番目の要素を取り出す
print(images.size())

G = Generator(z_dim=20, image_size=64)
D = Discriminator(z_dim=20, image_size=64)

#初期化の実施
G.apply(weights_init)
D.apply(weights_init)

print("ネットワークの初期化")

#学習・検証を実施する
#6分ほどかかる(GPUの場合)
num_epochs = 200
G_update, D_update = train_model(
        G, D, dataloader=train_dataloader, num_epochs=num_epochs)

#生成画像と訓練データを可視化する
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#入力の乱数生成
batch_size = 8
z_dim = 20
fixed_z = torch.randn(batch_size, z_dim)
fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

#画像生成
fake_images = G_update(fixed_z.to(device)) #イテレータに変換
imges = next(batch_iterator) #1番目の要素を取り出す

#出力
fig = plt.figure(figsize=(15, 6))
for i in range(0, 5):
    #上段に訓練データを
    plt.subplot(2, 5, i+1)
    plt.imshow(imges[i][0].cpu().detach(), 'gray')

    #下段に生成データを表示する
    plt.subplot(2, 5, 5+i+1)
    plt.imshow(fake_imges[i][0].cpu().detach().numpy(), 'gray')
