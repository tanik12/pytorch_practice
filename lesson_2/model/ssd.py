import numpy as np
import pandas as pd
from itertools import product
from math import sqrt

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

from vgg import make_vgg
from extra import make_extras
from l2norm import L2Norm
from loc_conf import make_loc_conf
from default_box import DBox

#SSDクラスを作成する
class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase #train or inferenceを指定
        self.num_classes = cfg["num_classes"] #クラス数=21

        #SSDのネットワークを作る
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
                cfg["num_classes"], cfg["bbox_aspect_num"])

        #DBoxの作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        #推論時はクラス[Detect]を用意します。
        if phase == "inference":
            self.detect = Detect()

if __name__ == "__main__":
    #動作確認
    #SSD300の設定
    ssd_cfg = {
            'num_classes': 21, #背景クラスを含めた合計クラス数
            'input_size': 300, #画像の入力サイズ
            'bbox_aspect_num': [4, 6, 6, 6, 4, 4], #出力するDBoxのアスペクト比の種類
            'feature_maps': [38, 19, 10, 5, 3, 1], #各sourceの画像サイズ
            'steps': [8, 16, 32, 64, 100, 300], #DBOXの大きさを決める
            'min_sizes': [30, 60, 111, 162, 213, 264], #DBOXの大きさを決める
            'max_sizes': [60, 111, 162, 213, 264, 315], #DBOXの大きさを決める
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }
    ssd_test = SSD(phase="train", cfg=ssd_cfg)
    print(ssd_test)
