import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.eval()

print(net)
