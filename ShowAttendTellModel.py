
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import argparse
import torch.nn.init as init

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
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.ss_prob = 0.0

        # Define encoder
        self.resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-3])
        # Replace last layer with image embedding layer
        #self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.embed_size)
        #self.bn = nn.BatchNorm1d(self.embed_size, momentum=0.01)
        self.finetune(allow=False)

        # Define decoder
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(2*self.embed_size, self.hidden_size, self.num_layers, batch_first=False)
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)

        # attention layer
        self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_att = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
        # image embed
        self.image_att_w = nn.Parameter(torch.FloatTensor(2*self.hidden_size, self.hidden_size))
        self.weight_ctc = nn.Parameter(torch.FloatTensor(2*self.hidden_size, self.hidden_size))
        # output layer
        self.out_weight = nn.Linear(self.hidden_size,self.hidden_size)
        self.init_weight()

    def forward(self, images, captions, maxlen=None, gt=True):
        # images : [batch x 3 x 224 x 224]
        # xt : [batch x embed_size], encode images with encoder,
        xt = self.encoder(images)
        xt = xt.view(xt.size(0), xt.size(1), -1).transpose(2, 1)  # [batch, 196, 1024]
        context_encode = torch.bmm(xt, self.image_att_w.unsqueeze(0).expand(xt.size(0), self.image_att_w.size(0), self.image_att_w.size(1)))  # [batch, 196, 512]
        # caption : [batch x seq x embed_size], embed captions with embeddings
        captions = self.embedding(captions)
        state = self.init_hidden(xt.size(0))
        # Sequence Length, we can manually designate maximum sequence length
        # or get maximum sequence length in ground truth captions
        seqlen = maxlen if maxlen is not None else captions.data.size(1)
        hidden, state = self.lstm(xt.mean(1).squeeze(1).unsqueeze(0), state)

        alpha_list = []
        hiddens = []
        outputs = []

        for t in range(seqlen):
            # One step over lstm cell
            embedding = captions[:, t, :]
            context, alpha = self.attention_layer(xt, context_encode, hidden)
            rnn_cat = torch.cat([context, embedding], dim=1)
            hidden, state = self.lstm(rnn_cat.unsqueeze(0), state)
            output = self.output_layer(context, hidden)

            alpha_list.append(alpha)
            outputs.append(output)
            hiddens.append(hidden)
        return outputs, seqlen


    def init_weight(self):
        """ initialize weight parameters """
        init.uniform(self.embedding.weight, a=-1.0, b=1.0)
        init.xavier_normal(self.classifier.weight.data, gain=np.sqrt(2.0))
        init.constant(self.classifier.bias.data, val=0)
        init.xavier_normal(self.weight_hh.weight.data, gain=np.sqrt(2.0))
        init.constant(self.weight_hh.bias.data, val=0)
        init.xavier_normal(self.out_weight.weight.data, gain=np.sqrt(2.0))
        init.constant(self.out_weight.bias.data, val=0)
        init.xavier_normal(self.weight_ctc.data, gain=np.sqrt(2.0))
        init.xavier_normal(self.image_att_w.data, gain=np.sqrt(2.0))
        init.xavier_normal(self.weight_att.data, gain=np.sqrt(2.0))

    def attention_layer(self, xt, context_encode, hidden):
        hidden = torch.squeeze(hidden)#.unsqueeze(1).expand_as(context_encode)
        h_att = F.tanh(context_encode + self.weight_hh(hidden).unsqueeze(1).expand_as(context_encode))  # [batch, 196, 512]
        out_att = torch.bmm(h_att, self.weight_att.unsqueeze(0).expand(h_att.size(0), self.weight_att.size(0), self.weight_att.size(1))).squeeze(2)  # [batch, 196]
        alpha = F.softmax(out_att)
        xt_emb = torch.bmm(xt, self.weight_ctc.unsqueeze(0).expand(xt.size(0), self.weight_ctc.size(0), self.weight_ctc.size(1)))
        context = (xt_emb * alpha.unsqueeze(2).expand_as(xt_emb)).mean(1).squeeze(1)
        return context, alpha

    def output_layer(self, context, hidden, prev=None):
        out = self.out_weight(context+hidden)
        out = self.classifier(F.tanh(out))
        return out

    def sample(self, images, maxlen=20):
        start_word = torch.ones(images.size(0))
        embeddings = self.embedding(Variable(start_word.long()).cuda())
        sample_ids = []

        xt = self.encoder(images)  # [batch, 1024, 14, 14]
        xt = xt.view(xt.size(0), xt.size(1), -1).transpose(2, 1)  # [batch, 196, 1024]
        context_encode = torch.bmm(xt, self.image_att_w.unsqueeze(0).expand(xt.size(0), self.image_att_w.size(0), self.image_att_w.size(1)))  # [batch, 196, 512]
        state = self.init_hidden(xt.size(0))
        hidden, state = self.lstm(xt.mean(1).squeeze(1).unsqueeze(0), state)
        outputs = []

        for t in range(maxlen):
            context, alpha = self.attention_layer(xt, context_encode, hidden)
            if t == 0:
                rnn_cat = torch.cat([embeddings, context], dim=1)
            else:
                rnn_cat = torch.cat([out_embed, context], dim=1)
            hidden, state = self.lstm(rnn_cat.unsqueeze(0), state)
            output = self.output_layer(context, hidden)
            predicted = output.max(1)[1]
            outputs.append(predicted)
            out_embed = self.embedding(predicted).squeeze(1)
        generated_sentence = torch.cat(outputs, 1)
        return generated_sentence.squeeze()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()))

    def encoder(self, images):
        # Extract the image feature vectors
        features = self.resnet(images)
        #features = self.bn(features)
        return features

    def finetune(self, allow=False):
        for param in self.resnet.parameters():
            param.requires_grad = True if allow else False
        #for param in self.resnet.fc.parameters():
        #    param.requires_grad = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=10000, help='dimension of word embedding vectors')
    parser.add_argument('--embed_size', type=int, default=1024, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()
    model = ShowAttendTellModel(args)
    model.cuda()
    xt = Variable(torch.ones([80, 3, 224, 224])).cuda()
    caption = Variable(torch.ones([80, 20])).long().cuda()
    d = model(xt, caption[:, :-1])