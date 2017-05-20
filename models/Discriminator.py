import torch
import torch.nn as nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()


    def forward(self, images, sentences):
        """"""