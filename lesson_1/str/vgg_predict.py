import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms

#preprocessing
class BaseTransform():
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.base_transform(img)

#maxid to label name
class ILSVRPredictor():
    def __init__(self, class_index):
        self.class_index = class_index
    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

img_path = '/Users/gisen/git/pytorch_advanced/1_image_classification/data/goldenretriever-3724972_640.jpg'

ILSVRC_class_index = json.load(open('/Users/gisen/git/pytorch_advanced/1_image_classification/data/imagenet_class_index.json', 'r'))

### preprocessing pram
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

### load img
predictor = ILSVRPredictor(ILSVRC_class_index)
image_file_path = img_path
img = Image.open(image_file_path)
###

### preprocessing and add dim ([3, 224, 224]⇨ [1, 3, 224, 224])
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)
inputs = img_transformed.unsqueeze_(0)
###

### model load and predict
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
out = net(inputs)
result = predictor.predict_max(out)
####

print("入力画像の予測: ", result)
