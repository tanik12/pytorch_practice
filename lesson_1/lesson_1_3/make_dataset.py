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

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    #epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-----------')
        #epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() #モデルを訓練モードに
            else:
                net.eval() #モデルを検証モードに

            epoch_loss = 0.0 #epochの損失和
            epoch_corrects = 0 #epochの正解数

            #未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            #データローダーからミニバッチを出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                #optimizerを初期化
                optimizer.zero_grad()
                #順伝搬(forward)計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels) #損失を計算
                    _, preds = torch.max(outputs, 1) #labelを予測
                    
                    #訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    #イテレーション結果の計算
                    #lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)
                    #正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)
                
                #epochごとのlossと正解率を表示
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

if __name__ == "__main__":
    ##########
    #trainとvalのdata_pathの指定
    train_list, rootpath_train = make_datapath_list(phase='train')
    val_list, _ = make_datapath_list(phase='val')
    ##########

    ##########
    #trainとvalのデータセットに分ける。前処理も含める。
    #pre-preprocessing img and show preprocessing img
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_dataset = HymenopteraDataset(
            file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = HymenopteraDataset(
            file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

    #動作確認
    index = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])
    ##########

    ##########
    #DataLoaderの作成
    batch_size = 32 #mini batch size
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size, shuffle=False)
    ##########

    #辞書型変数にまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
    #動作確認
    batch_iterator = iter(dataloaders_dict["train"]) #イテレーターに変換
    inputs, labels = next(
            batch_iterator)
    print(inputs.size())
    print(labels)
    ###########
    
    ###########
    #学習済みvgg16モデルをロード.vgg16モデルのインスタンスを生成
    use_pretrained = True #学習済みのパラメーターを使用
    net = models.vgg16(pretrained=use_pretrained)

    #vgg16の最後の出力層の出力ユニットをアリとハチの2つに付け替える
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    #訓練モードに設定
    net.train()

    print('ネットワーク設定完了:学習済みの重みをロードし、訓練モードに設定しました。')
    ###########

    ###########
    #損失関数の定義
    criterion = nn.CrossEntropyLoss()
    ###########

    ###########
    #最適化手法の設定
    #転移学習で学習させるパラメータを、変数params_to_updateに格納する.
    params_to_update = []
    #学習させるパラメータ名
    update_param_names = ["classifier.6.weight", "classifier.6.bias"]
    #学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
            print(name)
        else:
            param.requires_grad = False
    
    #params_to_updateの中身を確認
    print("--------")
    print(params_to_update)

    #最適化手法の設定
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
    ###########

    ###########
    #学習・検証を実行する
    num_epochs = 2
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
    ###########
