import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, opt):
        super(EncoderCNN, self).__init__()
        self.cnn = models.resnet34(pretrained=True) if opt.cnn_type == 'resnet' else models.vgg16(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, opt.img_embed_size)
        self.bn = nn.BatchNorm1d(opt.img_embed_size, momentum=0.01)
        self.finetune(allow=False)
        self.init_weights()

    def forward(self, images):
        features = self.cnn(images)
        features = self.bn(features)
        return features

    def init_weights(self):
        """ initialize weight parameter """
        self.cnn.fc.weight.data.normal_(0.0, 0.02)
        self.cnn.fc.bias.data.fill_(0)

    def finetune(self, allow=False):
        for param in self.cnn.parameters():
            param.requires_grad = True if allow else False
        for param in self.cnn.fc.parameters():
            param.requires_grad = True

class EncoderCNN_F(nn.Module):
    def __init__(self, opt):
        super(EncoderCNN_F, self).__init__()
        self.cnn = models.resnet101(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-3])
        self.finetune(allow=False)

    def forward(self, images):
        features = self.cnn(images)
        return features

    def finetune(self, allow=False):
        for param in self.cnn.parameters():
            param.requires_grad = True if allow else False
