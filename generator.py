import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class GeneratorModel(nn.Module):

    def __init__(self, hidden_size, context_size, vocab_size, embed_size, opt, feature_size=[196, 512]):
        super(GeneratorModel, self).__init__()
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
        self.weight_att= nn.Parameter(torch.FloatTensor(feature_size[1], 1))

        """ define decoder, use lstm cell for reproducing """
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstmcell = nn.LSTMCell(hidden_size , hidden_size)

        """ define classifier """
        self.context2out= nn.Linear(context_size, embed_size)
        self.hidden2tout= nn.Linear(hidden_size, embed_size)
        self.dropout = nn.Dropout(p=0.5)

        self.classifier = nn.Linear(embed_size, vocab_size)
        self.init_weight()

    def forward(self, images, captions, lengths):

        start_word = torch.ones(images.size(0))
        embeddings = self.embedding(Variable(start_word.long()).cuda())
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        features = self.encoder(images)  # [batch, 512, 14, 14]
        features = features.view(features.size(0), features.size(1), -1).transpose(2, 1)  # [batch, 196, 512]
        context_encode = torch.bmm(features, self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0), self.image_att_w.size(1)))  # [batch, 196, 512]

        hidden, c = self.init_lstm(features)

        for i in range(20):  # maximum sampling length
            context, alpha = self.attention_layer(features, context_encode, hidden)
            if i == 0:
                rnn_input = torch.cat([embeddings, context], dim=1)
            hidden, c = self.lstmcell(rnn_input, (hidden, c))  # (batch_size, 1, hidden_size)
            outputs = self.output_layer(context, hidden)  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            embedding = self.embedding(predicted).squeeze(1)
            rnn_input = torch.cat([embedding, context], dim=1)
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)

        return sampled_ids.squeeze()

    def init_weight(self):
        """"""
        """ initialize weight parameters """
        init.uniform(self.embedding.weight, a=-1.0, b=1.0)
        init.xavier_normal(self.image_att_w.data, gain=np.sqrt(2.0))
        init.xavier_normal(self.init_hidden.weight.data, gain=np.sqrt(2.0))
        init.constant(self.init_hidden.bias.data, val=0)
        init.xavier_normal(self.init_memory.weight.data, gain=np.sqrt(2.0))
        init.constant(self.init_memory.bias.data, val=0)
        init.xavier_normal(self.hidden2tout.weight.data, gain=np.sqrt(2.0))
        init.constant(self.hidden2tout.bias.data, val=0)
        init.xavier_normal(self.context2out.weight.data, gain=np.sqrt(2.0))
        init.constant(self.context2out.bias.data, val=0)
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

    def attention_layer(self, features, context_encode, hidden):
        h_att = F.tanh(context_encode + self.weight_hh(hidden).unsqueeze(1).expand_as(context_encode))
        out_att = torch.bmm(h_att, self.weight_att.unsqueeze(0).expand(h_att.size(0), self.weight_att.size(0), self.weight_att.size(1))).squeeze(2)
        alpha = F.softmax(out_att)
        context = (features * alpha.unsqueeze(2).expand_as(features)).mean(1).squeeze(1)
        return context, alpha

    def output_layer(self, context, hidden, prev=None):
        context = self.context2out(context)
        hidden = self.hidden2tout(hidden)
        out = self.classifier(context + hidden)

        return out

    def finetune(self, allow=False):
        for param in self.encoder.parameters():
            param.requires_grad = True if allow else False

    def sample(self, images, states):
        """"""
        start_word = torch.ones(images.size(0))
        embeddings = self.embedding(Variable(start_word.long()).cuda())
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        features = self.encoder(images)  # [batch, 512, 14, 14]
        features = features.view(features.size(0), features.size(1), -1).transpose(2, 1)  # [batch, 196, 512]
        context_encode = torch.bmm(features, self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0), self.image_att_w.size(1)))  # [batch, 196, 512]
        hidden, c = self.init_lstm(features)
        for i in range(20):  # maximum sampling length
            context, alpha = self.attention_layer(features, context_encode, hidden)
            if i == 0:
                rnn_input = torch.cat([embeddings, context], dim=1)
            hidden, c = self.lstmcell(rnn_input, (hidden, c))  # (batch_size, 1, hidden_size)
            outputs = self.output_layer(context, hidden)  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            embedding = self.embedding(predicted).squeeze(1)
            rnn_input = torch.cat([embedding, context], dim=1)
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)
        return sampled_ids.squeeze()

    def sample_beam(self, images, state, beam_size):
        """"""