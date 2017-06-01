
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

        # Define decoder
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=False)
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, features, captions, seqlen, gt=True):
        # features : [batch x embed_size], encode images with encoder,
        # caption : [batch x seq x embed_size], embed captions with embeddings
        captions = self.embedding(captions)
        state = self.init_hidden(features.size(0))

        hidden, state = self.lstm(features.unsqueeze(0), state)
        outputs = []
        # Loop for the sequence
        for t in range(seqlen):
            # One step over lstm cell
            xt = captions[:, t, :]
            hidden, state = self.lstm(xt.unsqueeze(0), state)
            output = self.classifier(hidden.squeeze(0))
            # end_output = self.classifier_end_token(hidden.squeeze(0))
            outputs.append(output)

        # outputs = torch.cat(outputs, 0)
        return outputs

    def sample(self, features, maxlen=20):

        batch_size = features.size(0)
        state = self.init_hidden(batch_size)

        hidden, state = self.lstm(features.unsqueeze(0), state)
        outputs = []
        word = Variable(torch.ones(batch_size).long()).cuda()
        xt = self.embedding(word)
        for t in range(maxlen):

            hidden, state = self.lstm(xt.unsqueeze(0), state)
            output = self.classifier(hidden.squeeze(0))
            predicted = output.max(1)[1]
            outputs.append(predicted)
            xt = self.embedding(predicted).squeeze(1)

        generated_sentence = torch.cat(outputs, 1)
        return generated_sentence.squeeze()

    def sample_reinforce(self, features, maxlen=20):

        batch_size = features.size(0)
        state = self.init_hidden(batch_size)

        hidden, state = self.lstm(features.unsqueeze(0), state)
        actions = []
        word = Variable(torch.ones(batch_size).long()).cuda()
        xt = self.embedding(word)
        for t in range(maxlen):

            hidden, state = self.lstm(xt.unsqueeze(0), state)
            output = self.classifier(hidden.squeeze(0))
            # Get a Stochastic Action (Stochastic Policy)
            action = self.getStochasticAction(output)
            actions.append(action)
            xt = self.embedding(action.detach()).squeeze(1)

        generated_sentence = torch.cat(actions, 1)
        return generated_sentence.squeeze(), actions

    def sample_gumble_softmax(self, features, maxlen=20):

        batch_size = features.size(0)
        state = self.init_hidden(batch_size)

        hidden, state = self.lstm(features.unsqueeze(0), state)
        actions = []
        word = Variable(torch.ones(batch_size).long()).cuda()
        xt = self.embedding(word)
        for t in range(maxlen):
            hidden, state = self.lstm(xt.unsqueeze(0), state)
            output = self.classifier(hidden.squeeze(0))
            # Get a Stochastic Action (Stochastic Policy)
            _, action = self.gumbel_softmax_sample(output)
            actions.append(action)
            xt = self.embedding(action.detach()).squeeze(1)

        generated_sentence = torch.cat(actions, 1)
        return generated_sentence.squeeze(), actions


    def getStochasticAction(self, output):
        distribution = F.softmax(output)
        action = distribution.multinomial()
        return action

    def sample_gumbel(self, input):
        noise = torch.rand(input.size())
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise).cuda()

    def gumbel_softmax_sample(self, input):
        temperature = 1
        noise = self.sample_gumbel(input)
        x = (input + noise) / temperature
        x = F.softmax(x)

        _, max_inx = torch.max(x, x.dim() - 1)
        x_hard = torch.cuda.FloatTensor(x.size()).zero_().scatter_(x.dim() - 1, max_inx.data, 1.0)
        x2 = x.clone()
        tmp = Variable(x_hard - x2.data)
        tmp.detach_()

        x = tmp + x

        return x.view_as(input), max_inx

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()))


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