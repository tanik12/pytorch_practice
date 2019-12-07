import matplotlib.pyplot as plt
from gan import Generator, Discriminator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

if __name__ == "__main__":
    #################
    ### Generator ###
    G = Generator(z_dim=20, image_size=64)

    #入力する乱数
    input_z = torch.randn(1, 20)

    #テンソルサイズを(1, 20, 1, 1)に変換
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

    #偽画像を出力
    fake_images = G(input_z)

    img_transformed = fake_images[0][0].detach().numpy()
    plt.imshow(img_transformed, 'gray')
    plt.show()
    #################

    #####################
    ### Discriminator ###
    #動作確認
    D = Discriminator(z_dim=20, image_size=64)

    #偽画像を生成
    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

    #偽画像をDに入力
    d_out = D(fake_images)

    #出力d_outにSigmoidをかけて0から1に変換
    print(nn.Sigmoid()(d_out))
    #####################

    
    #########################
    #########################
    # Dの誤差関数のイメージ実装
    # maximize log(D(x)) + log(1 - D(G(z)))
    #########################

    #正解ラベルを作成
    mini_batch_size = 2
    label_real = torch.full((mini_batch_size,), 1)
    
    #偽ラベルを作成
    label_fake = torch.full((mini_batch_size,), 0)

    #誤差関数を定義
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    #真の画像を判定
    d_out_real = D(x)

    #偽の画像を生成して判定
    input_z = torch.randn(mini_batch_size, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    fake_images = G(input_z)
    d_out_fake = D(fake_images)

    # 誤差を計算
    d_loss_real = criterion(d_out_real.view(-1), label_real)
    d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
    d_loss = d_loss_real + d_loss_fake
    #########################

    #########################
    #########################
    # Gの誤差関数のイメージ実装
    # maximize log(D(G(z)))
    #########################

    #偽の画像を生成して判定
    input_z = torch.randon(mini_batch_size, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    fake_images = G(input_z)
    d_out_fake = D(fake_images)

    #誤差を計算
    g_loss = criterion(d_out_fake.view(-1), label_real)
    #########################
