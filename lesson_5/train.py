import matplotlib.pyplot as plt
from gan import Generator, Discriminator
from data_loader import make_datapath_list, ImageTransform, GAN_img_Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import time

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        #BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_model(G, D, dataloader, num_epochs):
    #GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    #最適化手法の設定
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    #誤差関数を定義
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    #パラメータをハードコーディング
    z_dim = 20
    mini_batch_size = 64

    #ネットワークをGPUへ
    G.to(device)
    D.to(device)

    G.train() #モデルを訓練モードに
    D.train() #モデルを訓練モードに

    #ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    #画像の枚数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    #イテレーションカウンタをセット
    iteration = 1
    logs = []

    #epochのループ
    for epoch in range(num_epochs):
        #開始時刻を保存
        t_epoch_start = time.time()
        epoch_g_loss = 0.0 #epochの損失和
        epoch_d_loss = 0.0 #epochの損失和

        print('---------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('---------------')
        print(' (train) ')

        #データローダーからminibatchずつ取り出すループ
        for imges in dataloader:
            # ------------------
            # Discriminatorの学習
            # ------------------
            #minibatchのsizeが1だと、batchnormalizationでエラーになるので避ける
            if imges.size()[0] == 1:
                continue
            #GPUが使えるならGPUにデータを送る
            imges = imges.to(device)
        
            #正解ラベルと偽ラベルの作成
            #epochの最後のイテレーションはminibatchが少なくなる
            mini_batch_size = imges.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            #真の画像を判定
            d_out_real = D(imges)

            #偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            #誤差を計算
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            #バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # ------------------
            # Generatorの学習
            # ------------------
            #偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            #誤差を計算
            g_loss = criterion(d_out_fake.view(-1), label_real)

            #バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # ------------------
            # 記録
            # ------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        #epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_loss:{:.4f}'.format(epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
        print('timer: {:4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()
    
    return G, D

if __name__ == "__main__":
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




