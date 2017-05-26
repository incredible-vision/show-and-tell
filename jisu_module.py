import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class argMax_gumble(nn.Module):
    def __init__(self):
        super(argMax_gumble, self).__init__()

    def sample_gumbel(self, input):
        noise = torch.FloatTensor(torch.rand(input.size()))
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise)

    def forward(self, x):
        temperature = 1
        noise = self.sample_gumbel(x)
        g = torch.add(x, noise.cuda()).div(temperature)
        g = F.softmax(g)

        _, max_inx = torch.max(g, g.dim() - 1)
        g_hard = torch.FloatTensor(g.size()).zero_().cuda().scatter_(g.dim() - 1, max_inx.data, 1.0)
        g2 = g.clone()
        tmp = Variable(g_hard - g2.data)
        tmp.detach_()

        g = tmp + g

        return g.view_as(x)

class Hadamard(nn.Module):
    def __init__(self, embed_size):
        super(Hadamard, self).__init__()
        self.W1 = nn.Linear(embed_size, embed_size)
        self.W2 = nn.Linear(embed_size, embed_size)

    def forward(self, x1, x2):
        return torch.mul(self.W1(x1), self.W2(x2))

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt
        self.embed_size = opt.embed_size
        self.hadamard = Hadamard(self.embed_size)
        self.fc = nn.Sequential(nn.Linear(self.embed_size, 256), nn.ReLU(True), nn.Linear(256, 2))

        self.bi_lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.embed_size, bias=True,
                               batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(self.embed_size*2, opt.vocab_size)
    def forward(self, text, img):
        x_1 = self.Text2Img(text, img)
        x_2 = self.Text2Text(text)
        return x_1, x_2

    def Text2Img(self, text, img):
        x = [self.hadamard(c, img).unsqueeze(2) for c in text]
        x = torch.cat(x, 2)
        x = torch.sum(x, 2).squeeze()
        x = self.fc(x)
        return x

    def Text2Text(self, text):
        text = [i.unsqueeze(0) for i in text]  # [20 x (1, 128, 512)]
        text = torch.cat(text, 0)              # [20, 128, 512]
        output, _ = self.bi_lstm(text)
        output = output.contiguous().view(-1, self.embed_size * 2)
        output = self.decoder(output)
        output = output.view(20, -1, self.opt.vocab_size)
        output = torch.chunk(output, output.size(0))
        output = [output[c].squeeze() for c in range(len(output))]
        return output

class Discriminator_cat(nn.Module):
    def __init__(self, opt):
        super(Discriminator_cat, self).__init__()
        self.embed_size = opt.embed_size
        self.hadamard = Hadamard(self.embed_size)
        self.fc = nn.Sequential(nn.Linear(self.embed_size * 20, 1024), nn.ReLU(True), nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 2))

    def forward(self, text, img):
        x = [self.hadamard(c, img) for c in text]
        x = torch.cat(x, 1)
        # print x.size()
        # x = torch.sum(x, 2).squeeze()
        x = self.fc(x)
        return x
