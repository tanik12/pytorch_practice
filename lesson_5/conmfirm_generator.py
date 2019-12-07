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
