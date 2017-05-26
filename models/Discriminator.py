import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(opt.vocab_size, 256)
        # self.convs1 = [nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(K, opt.embed_size)) for K in [3, 4, 3]]
        self.img_embed = nn.Linear(512, 256)

        self.conv13 = nn.Conv2d(1, 64, (3, 256))
        self.conv14 = nn.Conv2d(1, 64, (4, 256))
        self.conv15 = nn.Conv2d(1, 64, (5, 256))
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(3 * 64, 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, sentences, features):
        """"""
        x = self.embed(sentences)  # (N,W,D)
        features = self.img_embed(features)
        x = torch.cat([x, features.unsqueeze(1)], 1)
        x = x.unsqueeze(1)  # (N,Ci,W,D)

        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        feat = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)

        x = self.dropout(feat)  # (N,len(Ks)*Co)
        logit = self.classifier(x)  # (N,C)
        return logit, feat
