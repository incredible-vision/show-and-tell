import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(opt.vocab_size, opt.embed_size)
        # self.convs1 = [nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(K, opt.embed_size)) for K in [3, 4, 3]]
        self.conv13 = nn.Conv2d(1, 128, (3, opt.embed_size))
        self.conv14 = nn.Conv2d(1, 128, (4, opt.embed_size))
        self.conv15 = nn.Conv2d(1, 128, (5, opt.embed_size))
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(3 * 128, 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, sentences):
        """"""
        x = self.embed(sentences)  # (N,W,D)

        # x = Variable(x)

        x = x.unsqueeze(1)  # (N,Ci,W,D)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        # x = torch.cat(x, 1)

        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)

        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.classifier(x)  # (N,C)
        return logit
