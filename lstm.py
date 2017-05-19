import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
import torchvision.transforms as transforms
import json
import pickle
import argparse

from data_loader import get_loader

MAX_TIME_STEP = 20

class ShowAttendTellModel(nn.Module):
    def __init__(self, hidden_size, context_size, vocab_size, embed_size, opt, feature_size=[196, 512]):
        super(ShowAttendTellModel, self).__init__()
        """ define encoder, use resnet50 for reproducing """
        self.opt = opt
        self.encoder = vgg16(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.features)[:-3])
        self.finetune(allow=False)

        """ define weight parameters """
        self.image_att_w = nn.Parameter(torch.FloatTensor(feature_size[1], feature_size[1]))
        self.init_hidden = nn.Linear(feature_size[1], hidden_size, bias=True)
        self.init_memory = nn.Linear(feature_size[1], hidden_size, bias=True)

        self.weight_hh = nn.Linear(hidden_size, context_size)
        self.weight_att = nn.Parameter(torch.FloatTensor(feature_size[1], 1))

        """ define decoder, use lstm cell for reproducing """
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstmcell = nn.LSTMCell(hidden_size, hidden_size)

        """ define classifier """
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions, target_masks):

        embeddings = self.embedding(captions)

        """ put input data through cnn """
        features = self.encoder(images)  # [batch, 512, 14, 14]
        features = features.view(features.size(0), features.size(1), -1).transpose(2, 1)  # [batch, 196, 512]
        context_encode = torch.bmm(features,
                                   self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0),
                                                                        self.image_att_w.size(1)))  # [batch, 196, 512]

        """ initialize hidden and memory unit"""
        hidden, c = self.init_lstm(features)

        alpha_list = []
        hiddens = []
        for t in range(MAX_TIME_STEP):
            embedding = embeddings[:, t, :]
            context, alpha = self.attention_layer(features, context_encode, hidden)
            rnn_input = torch.cat([embedding, context], dim=1)
            hidden, c = self.lstmcell(rnn_input, (hidden, c))
            alpha_list.append(alpha)
            for i, index in enumerate(torch.transpose(target_masks, 0, 1)[t]):
                if index.data[0] == 1:
                    hiddens.append(hidden[i])
        return hiddens

    def init_lstm(self, features):
        features_mean = features.mean(1).squeeze(1)
        h = self.init_hidden(features_mean)
        c = self.init_memory(features_mean)
        return h, c

    def attention_layer(self, features, context_encode, hidden):
        h_att = F.tanh(context_encode + self.weight_hh(hidden).unsqueeze(1).expand_as(context_encode))
        out_att = torch.bmm(h_att, self.weight_att.unsqueeze(0).expand(h_att.size(0), self.weight_att.size(0),
                                                                       self.weight_att.size(1))).squeeze(2)
        alpha = F.softmax(out_att)
        context = (features * alpha.unsqueeze(2).expand_as(features)).mean(1).squeeze(1)
        return context, alpha

    def output_layer(self, context, hidden):

        return None

    def finetune(self, allow=False):
        for param in self.encoder.parameters():
            param.requires_grad = True if allow else False


model = ShowAttendTellModel(hidden_size=1024, context_size=512, vocab_size=10000, embed_size=512, opt=1).cuda()
data = Variable(torch.FloatTensor(torch.rand([10, 3, 224, 224]))).cuda()
caption = Variable(torch.ones([10, 20])).long().cuda()
target_masks = Variable(torch.cat((torch.ones([5, 20]), torch.zeros([5, 20])), 0)).long().cuda()
res = model(data, caption, target_masks)

sorted, indices = torch.sort(x)

if __name__ == "__main__":
    """"""