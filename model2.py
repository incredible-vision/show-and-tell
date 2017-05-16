import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


class ShowAttendTellModel(nn.Module):
    def __init__(self, hidden_size, context_size, vocab_size, embed_size, opt, feature_size=[196, 512]):
        super(ShowAttendTellModel, self).__init__()
        """ define encoder, use resnet50 for reproducing """
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.encoder = vgg16(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.features)[:-3])
        self.finetune(allow=False)

        """ define weight parameters """
        self.image_att_w = nn.Parameter(torch.FloatTensor(feature_size[1], hidden_size))
        # self.init_hidden = nn.Linear(feature_size[1], hidden_size, bias=True)
        # self.init_memory = nn.Linear(feature_size[1], hidden_size, bias=True)

        self.weight_hh = nn.Linear(hidden_size, hidden_size)
        self.weight_att = nn.Parameter(torch.FloatTensor(hidden_size, 1))

        self.weight_gatex = nn.Linear(embed_size, hidden_size)
        self.weight_gateh = nn.Linear(hidden_size, hidden_size)

        self.weight_atts = nn.Linear(hidden_size, hidden_size)
        self.sentinelout = nn.Linear(hidden_size, 1)

        """ define decoder, use lstm cell for reproducing """
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstmcell = nn.LSTMCell(embed_size, hidden_size)

        """ define classifier """
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(embed_size, vocab_size)
        self.init_weight()

    def forward(self, images, captions, lengths):

        embeddings = self.embedding(captions)
        packed, batch_sizes = pack_padded_sequence(embeddings, lengths, batch_first=True)
        """ put input data through cnn """
        features = self.encoder(images)  # [batch, 512, 14, 14]
        features = features.view(features.size(0), features.size(1), -1).transpose(2, 1)  # [batch, 196, 512]
        context_encode = torch.bmm(features,
                                   self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0),
                                                                        self.image_att_w.size(1)))  # [batch, 196, 512]

        """ initialize hidden and memory unit"""
        hidden, c = self.initialize_hidden(features.size(0))

        alpha_list = []
        hiddens = []
        outputs = []

        for t, batch_size in enumerate(batch_sizes):
            embedding = embeddings[:batch_size, t, :]
            rnn_input = embedding
            g, s = self.sentinel(rnn_input, hidden[:batch_size], c[:batch_size])
            hidden, c = self.lstmcell(rnn_input, (hidden[:batch_size], c[:batch_size]))
            context, alpha = self.attention_layer(features[:batch_size], context_encode[:batch_size], hidden, s)
            # hidden = self.dropout(hidden)
            output = self.output_layer(context, hidden)
            alpha_list.append(alpha)
            hiddens.append(hidden)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        return outputs

    def init_weight(self):
        """"""
        """ initialize weight parameters """
        init.uniform(self.embedding.weight, a=-1.0, b=1.0)
        init.xavier_normal(self.image_att_w.data, gain=np.sqrt(2.0))
        # init.xavier_normal(self.init_hidden.weight.data, gain=np.sqrt(2.0))
        # init.constant(self.init_hidden.bias.data, val=0)
        # init.xavier_normal(self.init_memory.weight.data, gain=np.sqrt(2.0))
        # init.constant(self.init_memory.bias.data, val=0)
        #init.xavier_normal(self.hidden2tout.weight.data, gain=np.sqrt(2.0))
        #init.constant(self.hidden2tout.bias.data, val=0)
        # init.xavier_normal(self.context2out.weight.data, gain=np.sqrt(2.0))
        # init.constant(self.context2out.bias.data, val=0)
        init.xavier_normal(self.weight_att.data, gain=np.sqrt(2.0))
        init.xavier_normal(self.weight_hh.weight.data, gain=np.sqrt(2.0))
        init.constant(self.weight_hh.bias.data, val=0)
        init.xavier_normal(self.classifier.weight.data, gain=np.sqrt(2.0))
        init.constant(self.classifier.bias.data, val=0)
        init.xavier_normal(self.lstmcell.weight_hh.data, gain=np.sqrt(2.0))
        init.xavier_normal(self.lstmcell.weight_ih.data, gain=np.sqrt(2.0))
        init.constant(self.lstmcell.bias_hh.data, val=0)
        init.constant(self.lstmcell.bias_ih.data, val=0)

    def init_lstm(self, features):
        features_mean = features.mean(1).squeeze(1)
        h = self.init_hidden(features_mean)
        c = self.init_memory(features_mean)
        return h, c

    def initialize_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(batch_size, self.hidden_size).zero_()),
                Variable(weight.new(batch_size, self.hidden_size).zero_()))

    def sentinel(self, embedding, hidden, c):
        g = F.sigmoid(self.weight_gateh(hidden) + self.weight_gatex(embedding))
        s = g * F.tanh(c)
        return g, s

    def attention_layer(self, features, context_encode, hidden, s):
        h_att = F.tanh(context_encode + self.weight_hh(hidden).unsqueeze(1).expand_as(context_encode))
        out_att_z = torch.bmm(h_att, self.weight_att.unsqueeze(0).expand(h_att.size(0), self.weight_att.size(0),
                                                                       self.weight_att.size(1))).squeeze(2)
        s_att = F.tanh(self.weight_atts(s) + self.weight_hh(hidden))
        out_att_s = self.sentinelout(s_att)
        alpha = F.softmax(torch.cat([out_att_z, out_att_s], dim=1))
        beta = alpha[:, -1]
        context = (features * alpha[:, :-1].unsqueeze(2).expand_as(features)).mean(1).squeeze(1)

        beta = beta.unsqueeze(1).expand(hidden.size(0), 512)
        a = beta*s
        b = 1-beta
        context = a + b*context
        return context, alpha

    def output_layer(self, context, hidden, prev=None):
        # context = self.context2out(context)
        #hidden = self.hidden2tout(hidden)
        out = self.dropout(F.tanh(context + hidden))
        out = self.classifier(out)
        return out

    def finetune(self, allow=False):
        for param in self.encoder.parameters():
            param.requires_grad = True if allow else False

    def sample(self, images):
        """"""
        start_word = torch.ones(images.size(0))
        embeddings = self.embedding(Variable(start_word.long()).cuda())
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        features = self.encoder(images)  # [batch, 512, 14, 14]
        features = features.view(features.size(0), features.size(1), -1).transpose(2, 1)  # [batch, 196, 512]
        context_encode = torch.bmm(features,
                                   self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0),
                                                                        self.image_att_w.size(1)))  # [batch, 196, 512]
        hidden, c = self.initialize_hidden(features.size(0))
        for i in range(20):  # maximum sampling length

            if i == 0:
                rnn_input = embeddings

            g, s = self.sentinel(rnn_input, hidden, c)
            hidden, c = self.lstmcell(rnn_input, (hidden, c))
            context, alpha = self.attention_layer(features, context_encode, hidden, s)
            # hidden = self.dropout(hidden)
            outputs = self.output_layer(context, hidden)

            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            rnn_input = self.embedding(predicted).squeeze(1)

        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)
        return sampled_ids.squeeze()

    def sample_beam(self, images, state, beam_size):
        """"""