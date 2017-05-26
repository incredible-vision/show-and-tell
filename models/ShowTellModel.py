
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import argparse

class LSTMCustom(nn.Module):
    def __init__(self, embed_size, hidden_size, rnn_dropout):
        super(LSTMCustom, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = rnn_dropout

        # Build Custom LSTM
        self.W_ix = nn.Linear(self.embed_size, self.hidden_size)
        self.W_ih = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_fx = nn.Linear(self.embed_size, self.hidden_size)
        self.W_fh = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_ox = nn.Linear(self.embed_size, self.hidden_size)
        self.W_oh = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_cx = nn.Linear(self.embed_size, self.hidden_size)
        self.W_ch = nn.Linear(self.hidden_size, self.hidden_size)

        self.rnn_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, xt, state):

        h, c = state

        i_gate = F.sigmoid(self.W_ix(xt) + self.W_ih(h))
        f_gate = F.sigmoid(self.W_fx(xt) + self.W_fh(h))
        o_gate = F.sigmoid(self.W_ox(xt) + self.W_oh(h))

        c = f_gate * c + i_gate * F.tanh(self.W_cx(xt) + self.W_ch(h))
        h = o_gate * c

        return h, (h, c)


# lstmcore = LSTMCustom(256, 256, 0.5)
# xt = Variable(torch.zeros([128, 256]))
# state = (Variable(torch.zeros(128, 256)),
#         Variable(torch.zeros(128, 256)))
# res = lstmcore(xt, state)


class ShowTellModel(nn.Module):
    """"" Implementation of Show and Tell Model for Image Captioning """""
    def __init__(self, opt):
        super(ShowTellModel, self).__init__()
        # Load hyper-parameters
        self.vocab_size = opt.vocab_size
        self.embed_size = opt.embed_size
        self.hidden_size= opt.hidden_size
        self.num_layers = opt.num_layers
        self.ss_prob = 0.0

        # Define encoder
        self.resnet = models.resnet101(pretrained=True)
        # Replace last layer with image embedding layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.embed_size)
        self.bn = nn.BatchNorm1d(self.embed_size, momentum=0.01)
        self.finetune(allow=False)

        # Define decoder
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=False)
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, images, captions, seqlen, gt=True):
        # images : [batch x 3 x 224 x 224]
        # xt : [batch x embed_size], encode images with encoder,
        xt = self.encoder(images)

        # caption : [batch x seq x embed_size], embed captions with embeddings
        captions = self.embedding(captions)
        state = self.init_hidden(xt.size(0))

        hidden, state = self.lstm(xt.unsqueeze(0), state)
        outputs = []
        # Loop for the sequence
        for t in range(seqlen):
            # One step over lstm cell
            xt = captions[:, t, :]
            hidden, state = self.lstm(xt.unsqueeze(0), state)
            output = self.classifier(hidden.squeeze(0))
            outputs.append(output)

        # outputs = torch.cat(outputs, 0)
        return outputs

    def sample(self, images, maxlen=20):

        xt = self.encoder(images)
        state = self.init_hidden(xt.size(0))

        hidden, state = self.lstm(xt.unsqueeze(0), state)
        outputs = []
        word = Variable(torch.ones(images.size(0)).long()).cuda()
        xt = self.embedding(word)
        for t in range(maxlen):

            hidden, state = self.lstm(xt.unsqueeze(0), state)
            output = self.classifier(hidden.squeeze(0))
            predicted = output.max(1)[1]
            outputs.append(predicted)
            xt = self.embedding(predicted).squeeze(1)

        generated_sentence = torch.cat(outputs, 1)
        return generated_sentence.squeeze()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()))

    def encoder(self, images):
        # Extract the image feature vectors
        features = self.resnet(images)
        features = self.bn(features)
        return features

    def finetune(self, allow=False):
        for param in self.resnet.parameters():
            param.requires_grad = True if allow else False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=10000, help='dimension of word embedding vectors')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()
    model = ShowTellModel(args)
    model.cuda()
    xt = Variable(torch.ones([128, 3, 224, 224])).cuda()
    caption = Variable(torch.ones([128, 20])).long().cuda()
    d = model(xt, caption[:,:-1])