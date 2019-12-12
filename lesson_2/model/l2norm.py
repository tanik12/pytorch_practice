import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import sys
sys.path.append('../')

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords,\
     PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

#convC4_3からの出力をscale=20のL2Normで正規化する層
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__() #親クラスのコンストラクタ実行
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale #係数weightの初期値として設定する値
        self.reset_parameters() #パラメータの初期化
        self.eps = 1e-10

    def reset_parameters(self):
        ''' 結合パラメータを大きさscaleの値にする初期化を実行 '''
        nn.init.constant_(self.weight, self.scale) #weightの値が全てscale(=20)になる

    def forward(self, x):
        '''
        38×38の特徴量に対して、512チャネルに渡って2乗和のルートを求めた38×38個の値を使用し、
        各特徴量を正規化してから係数を掛け算する層
        '''

        #各チャネルにおける38×38個の特徴量のチャネル方向の二乗和を計算し、
        #さらにルートを求め、割り算して正規化する。
        #normのテンソルサイズはtorch.Size([batch_num, 1, 38, 38])になる。
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        #係数をかける。係数はチャネルごとに１つで、512個の係数を持つ
        #self.weightのテンソルサイズはtorch.Size([512])なので
        #torch.Size([batch_num, 512, 38, 38])まで変形します
        #unsqueezeは次元を増やす関数。expand_as(x)はxと同じサイズにするという関数.
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        print("AAA : ", weights.size)
        out = weights * x

        return out
