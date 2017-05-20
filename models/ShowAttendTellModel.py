
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

class ShowAttendTellModel(nn.Module):
    """"" Implementation of Show and Tell Model for Image Captioning """""
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__()
        # Load hyper-parameters
        self.vocab_size = opt.vocab_size
        self.embed_size = opt.embed_size
        self.hidden_size= opt.hidden_size
        self.num_layers = opt.num_layers
        self.ss_prob = 0.0

        # Define decoder
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size * 2, self.hidden_size, self.num_layers, batch_first=False)
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)

        # attention layer
        self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_att = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
        # image embed
        self.image_att_w = nn.Parameter(torch.FloatTensor(2 * self.hidden_size, self.hidden_size))
        self.weight_ctc = nn.Parameter(torch.FloatTensor(2 * self.hidden_size, self.hidden_size))
        # output layer
        self.out_weight = nn.Linear(self.hidden_size, self.hidden_size)

        self.init_hidden = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
        self.init_memory = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)

    def forward(self, features, captions, seqlen, gt=True):
        # xt : [batch x embed_size], encode images with encoder,
        # caption : [batch x seq x embed_size], embed captions with embeddings
        image_encode = features.view(features.size(0), features.size(1), -1).transpose(2, 1) # [batch x 196 x 1024]
        context_encode = torch.bmm(image_encode, self.image_att_w.unsqueeze(0).expand(image_encode.size(0), self.image_att_w.size(0), self.image_att_w.size(1)))  # [batch, 196, 512]

        captions = self.embedding(captions)
        hidden, state = self.init_lstm(image_encode.mean(1).squeeze())

        outputs = []
        # Loop for the sequence
        for t in range(seqlen):
            # One step over lstm cell
            xt = captions[:, t, :]
            context, alpha = self.attention_layer(image_encode, context_encode, hidden)
            hidden, state = self.lstm(torch.cat([context, xt], 1).unsqueeze(0), state)
            output = self.output_layer(context, hidden.squeeze(0))
            outputs.append(output)

        # outputs = torch.cat(outputs, 0)
        return outputs

    def attention_layer(self, xt, context_encode, hidden):
        hidden = torch.squeeze(hidden)  # .unsqueeze(1).expand_as(context_encode)
        h_att = F.tanh(context_encode + self.weight_hh(hidden).unsqueeze(1).expand_as(context_encode))  # [batch, 196, 512]
        out_att = torch.bmm(h_att, self.weight_att.unsqueeze(0).expand(h_att.size(0), self.weight_att.size(0),
                                                                       self.weight_att.size(1))).squeeze(2)  # [batch, 196]
        alpha = F.softmax(out_att)
        xt_emb = torch.bmm(xt, self.weight_ctc.unsqueeze(0).expand(xt.size(0), self.weight_ctc.size(0),
                                                                   self.weight_ctc.size(1)))
        context = (xt_emb * alpha.unsqueeze(2).expand_as(xt_emb)).mean(1).squeeze(1)
        return context, alpha

    def output_layer(self, context, hidden, prev_word=None):
        out = context + hidden + prev_word if prev_word else (context + hidden)
        out = self.out_weight(out)
        out = self.classifier(F.tanh(out))
        return out

    def sample(self, features, maxlen=20):

        image_encode = features.view(features.size(0), features.size(1), -1).transpose(2, 1)  # [batch x 196 x 1024]
        context_encode = torch.bmm(image_encode, self.image_att_w.unsqueeze(0).expand(image_encode.size(0), self.image_att_w.size(0),
                                                                        self.image_att_w.size(1)))  # [batch, 196, 512]

        batch_size = features.size(0)
        hidden, state = self.init_lstm(image_encode.mean(1).squeeze())

        outputs = []
        word = Variable(torch.ones(batch_size).long()).cuda()
        xt = self.embedding(word)
        for t in range(maxlen):
            context, alpha = self.attention_layer(image_encode, context_encode, hidden)
            hidden, state = self.lstm(torch.cat([context, xt], 1).unsqueeze(0), state)
            output = self.output_layer(context, hidden.squeeze(0))
            predicted = output.max(1)[1]
            outputs.append(predicted)
            xt = self.embedding(predicted).squeeze(1)

        generated_sentence = torch.cat(outputs, 1)
        return generated_sentence.squeeze()

    def init_lstm(self, features):
        hidden = self.init_hidden(features).unsqueeze(0)
        c = self.init_memory(features).unsqueeze(0)
        return hidden, (hidden, c)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=10000, help='dimension of word embedding vectors')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()
    model = ShowAttendTellModel(args)
    model.cuda()
    xt = Variable(torch.ones([128, 3, 224, 224])).cuda()
    caption = Variable(torch.ones([128, 20])).long().cuda()
    d = model(xt, caption[:,:-1])