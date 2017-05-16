import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
import math
from torch.nn.utils.rnn import pack_padded_sequence


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
        self.weight_att= nn.Parameter(torch.FloatTensor(feature_size[1], 1))

        """ define decoder, use lstm cell for reproducing """
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstmcell = nn.LSTMCell(hidden_size, hidden_size)

        """ define output MLP for word selecting"""
        self.mlp_w = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        self.linear = nn.Linear(embed_size, vocab_size, bias=True)
        self.w_context = nn.Linear(context_size, embed_size)
        self.w_hidden = nn.Linear(hidden_size, embed_size)


    def forward(self, images, captions, lengths):
        """ put input data through cnn """
        features = self.encoder(images) # [batch, 512, 14, 14]
        features = features.view(features.size(0), features.size(1), -1).transpose(2, 1) # [batch, 196, 512]
        context_encode = torch.bmm(features, self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0), self.image_att_w.size(1))) # [batch, 196, 512]
        hidden, c = self.init_lstm(features)
        embeddings = self.embed(captions)
        packed, batch_sizes = pack_padded_sequence(embeddings, lengths, batch_first=True)

        alpha_list = []
        out = []
        hiddens = []

        for t, batch_size in enumerate(batch_sizes):
            embedding = embeddings[:batch_size, t, :]
            context, alpha = self.attention_layer(features[:batch_size], context_encode[:batch_size], hidden[:batch_size])
            embedding = torch.cat([embedding, context], 1)
            hidden, c = self.lstmcell(embedding, (hidden[:batch_size], c[:batch_size]))
            alpha_list.append(alpha)
            out.append(self.out_select(context, hidden))
            hiddens.append(hidden)
        out = torch.cat(out, dim=0)
        print(len(features))

        return features, out

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

    def out_select(self, context, hidden):
        context = self.w_context(context)
        hidden = self.w_hidden(hidden)
        out = self.linear(hidden+context)
        out = F.softmax(out)
        return out

    def finetune(self, allow=False):
        for param in self.encoder.parameters():
            param.requires_grad = True if allow else False

model = ShowAttendTellModel(hidden_size=1024, context_size=512, vocab_size=10000, embed_size=512, opt=1).cuda()
data = Variable(torch.FloatTensor(torch.rand([10, 3, 224, 224]))).cuda()
caption = Variable(torch.ones([10, 20])).long().cuda()
res = model(data, caption, [10, 9, 7, 6, 4, 3, 3, 1, 1, 1])

if __name__ == "__main__":
    """"""




