import sys
import random
import json
import string
import os
import pickle
import numpy as np

sys.path.append("coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

class ShowAttendTellModel_XE(nn.Module):
    """"" Implementation of Show and Tell Model for Image Captioning """""
    def __init__(self, opt):
        super(ShowAttendTellModel_XE, self).__init__()
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

    def forward(self, images, captions, maxlen=None, gt=True):
        # images : [batch x 3 x 224 x 224]
        # xt : [batch x embed_size], encode images with encoder,
        xt = self.encoder(images)
        # caption : [batch x seq x embed_size], embed captions with embeddings
        captions = self.embedding(captions)
        state = self.init_hidden(xt.size(0))
        # Sequence Length, we can manually designate maximum sequence length
        # or get maximum sequence length in ground truth captions
        seqlen = maxlen if maxlen is not None else captions.data.size(1)
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


# My ver.. stop making... -> use SE ver
class ShowAttendTellModel_PG(nn.Module):

    def __init__(self, hidden_size, context_size, vocab_size, embed_size, opt, feature_size=[196, 512]): # vgg:[196,512] res50:[196,1024]
        super(ShowAttendTellModel_PG, self).__init__()
        """ define encoder, use resnet50 for reproducing """
        self.opt = opt
        self.encoder = vgg16(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.features)[:-3]) # // if vgg16
        #elf.encoder = nn.Sequential(*list(self.encoder.children())[:-3]) # // if resnet50
        self.finetune(allow=False)

        """ define weight parameters """
        self.image_att_w = nn.Parameter(torch.FloatTensor(feature_size[1], feature_size[1]))
        self.init_hidden = nn.Linear(feature_size[1], hidden_size, bias=True)
        self.init_memory = nn.Linear(feature_size[1], hidden_size, bias=True)

        self.weight_hh = nn.Linear(hidden_size, context_size)
        self.weight_att= nn.Parameter(torch.FloatTensor(feature_size[1], 1))

        """ define decoder, use lstm cell for reproducing """
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstmcell = nn.LSTMCell(hidden_size, hidden_size)

        """ define classifier """
        self.context2out= nn.Linear(context_size, embed_size)
        self.hidden2tout= nn.Linear(hidden_size, embed_size)
        self.classifier = nn.Linear(embed_size, vocab_size)

        """ for PG """
        with open(opt.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        self.annFile = '/home/gt/D_Data/COCO/annotations_captions/captions_train2014.json'
        self.coco = COCO(self.annFile)
        self.valid = self.coco.getImgIds()


    def forward(self, images, captions, lengths, imgids, vocab):

        embeddings = self.embedding(captions)
        packed, batch_sizes = pack_padded_sequence(embeddings, lengths, batch_first=True)
        """ put input data through cnn """
        features = self.encoder(images) # [batch, 512, 14, 14]
        features = features.view(features.size(0), features.size(1), -1).transpose(2, 1) # vgg16:[batch, 196, 512] // res50:[batch, 196, 1024]
        context_encode = torch.bmm(features, self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0), self.image_att_w.size(1))) # [batch, 196, 512]

        """ initialize hidden and memory unit"""
        hidden, c = self.init_lstm(features)
        sample_init_state = [hidden, c]

        alpha_list = []
        hiddens = []
        outputs = []
        actions = []

        for t, batch_size in enumerate(batch_sizes):
            embedding = embeddings[:batch_size, t, :]
            context, alpha = self.attention_layer(features[:batch_size], context_encode[:batch_size], hidden[:batch_size])
            rnn_input = torch.cat([embedding, context], dim=1)
            hidden, c = self.lstmcell(rnn_input, (hidden[:batch_size], c[:batch_size]))
            output = self.output_layer(context, hidden)
            action = output.multinomial()

            alpha_list.append(alpha)
            hiddens.append(hidden)
            outputs.append(output)
            actions.append(action)

        #outputs = torch.cat(outputs, dim=0) # e.g [27, 10372] when (batch=2, length=[15,12])
        actions = torch.cat(actions, dim=0) # e.g [27, 1]     when (batch=2, length=[15,12])

        Q_s = self.generate_rewards(actions, imgids, batch_sizes, images, lengths)
        #Q_s = self.sorting_Q_s(Q_s, lengths) # ...ing

        actions = torch.split(actions, 1, dim=0) # [ 27 x (1x1 torch variable)]

        return actions, Q_s

    def sorting_Q_s(self, Q_s, lengths):
        ## Not yet .. ing
        packed, batch_sizes = pack_padded_sequence(Q_s, [3,3], batch_first=True)
        return Q_s


    def generate_rewards(self, actions_s, imgids, batch_sizes, images, lengths):

        unpacked_actions_s = pad_packed_sequence([actions_s, batch_sizes]) # 15, 2, 10000, [2,2,2... 1,1,1..]
        unpacked_actions_s = unpacked_actions_s[0].transpose(0, 1) # 2, 13, 10000
        Q_s = list()

        for batch_i, (actions, imgid, image) in enumerate(zip(unpacked_actions_s, imgids, images)):

            # embeddings = self.embedding(Variable(torch.zeros(image.size(0))).long().cuda())
            embeddings = self.embedding(Variable(torch.zeros(1)).long().cuda()) # batch 1
            features = self.encoder(image.unsqueeze(0))  # [batch, 512, 14, 14]
            features = features.view(features.size(0), features.size(1), -1).transpose(2, 1)  # [batch, 196, 512]
            context_encode = torch.bmm(features, self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0), self.image_att_w.size(1)))
            hidden, c = self.init_lstm(features)
            sample_info = {'embeddings':embeddings, 'features':features, 'context_encode':context_encode, 'hidden':hidden, 'c':c}

            step_reward_s = list()
            # Roll out
            for time_step in range(3): #20
                step_reward = 0.0
                for k in range(3):
                    sentence = self.rollout_sentence(sample_info, actions, time_step)
                    pred = {'image_id': imgid, 'caption': sentence}
                    reward = self.eval_reward_metrics(pred, self.coco)
                    step_reward += reward
                step_reward_s.append(step_reward/3)

            Q_s.append(step_reward_s)

        return Q_s


    def rollout_sentence(self, sample_info, actions, time_step):

        sentence = list()
        for i in range(20):  # maximum sampling length
            context, alpha = self.attention_layer(sample_info['features'], sample_info['context_encode'], sample_info['hidden'])
            if i == 0:
                rnn_input = torch.cat([sample_info['embeddings'], context], dim=1)

            hidden, c = self.lstmcell(rnn_input, (sample_info['hidden'], sample_info['c']))  # (batch_size, 1, hidden_size)
            outputs = self.output_layer(context, hidden)  # (batch_size, vocab_size)

            if i < time_step:
                predicted_word_idx = actions.data.cpu().numpy()[i][0]
                predicted_word = self.vocab.idx2word[predicted_word_idx]
                sentence.append(predicted_word)

            if i >= time_step:
                predicted_word_idx = outputs.squeeze(0).multinomial().data.cpu().numpy()[0]
                predicted_word = self.vocab.idx2word[predicted_word_idx]
                if predicted_word == '<end>': break
                sentence.append(predicted_word)

            _, embedding = outputs.max(1)
            embedding = self.embedding(embedding).squeeze(1)
            rnn_input = torch.cat([embedding, context], dim=1)

        sentence = ' '.join(sentence)

        return sentence


    def eval_reward_metrics(self, pred, coco):

        tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
        preds_filt = [pred]
        json.dump(preds_filt, open('cache/' + tmp_name + '.json', 'w'))
        resFile = 'cache/' + tmp_name + '.json'

        cocoRes = coco.loadRes(resFile)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
        os.system('rm ' + tmp_name + '.json')

        BMCR = 0.0
        for metric, score in cocoEval.eval.items():
            if metric == 'BLEU-1': BMCR += score * 0.5
            if metric == 'BLEU-2': BMCR += score * 0.5
            if metric == 'BLEU-3': BMCR += score * 1.0
            if metric == 'BLEU-4': BMCR += score * 1.0
            elif metric == 'CIDEr': BMCR += score * 1.0
            elif metric == 'METEOR': BMCR += score * 5.0
            elif metric == 'ROUGH': BMCR += score * 2.0

        return BMCR


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
        embeddings = self.embedding(Variable(torch.zeros(images.size(0))).long().cuda())
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        features = self.encoder(images)  # [batch, 512, 14, 14]
        features = features.view(features.size(0), features.size(1), -1).transpose(2, 1)  # [batch, 196, 512]
        context_encode = torch.bmm(features, self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0), self.image_att_w.size(1)))  # [batch, 196, 512]
        hidden , c = states
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


# test
class ShowAttendTellModel_G(nn.Module):

    def __init__(self, hidden_size, context_size, vocab_size, embed_size, opt, feature_size=[196, 512]):
        #                  1024         512         len()        512         vgg:[196,512] res50:[196,1024]
        super(ShowAttendTellModel_G, self).__init__()
        """ define encoder, use resnet50 for reproducing """
        self.opt = opt
        self.encoder = vgg16(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.features)[:-3])
        self.finetune(allow=False)

        """ define weight parameters """
        self.image_att_w = nn.Parameter(torch.FloatTensor(feature_size[1], feature_size[1])) # 512x512
        self.init_hidden = nn.Linear(feature_size[1], hidden_size, bias=True)                # 512x1024
        self.init_memory = nn.Linear(feature_size[1], hidden_size, bias=True)                # 512x1024

        self.weight_hh = nn.Linear(hidden_size, context_size)                                # 1024x512
        self.weight_att = nn.Parameter(torch.FloatTensor(feature_size[1], 1))                # 512x1

        """ define decoder, use lstm cell for reproducing """
        self.embedding = nn.Embedding(vocab_size, embed_size)                                # 8000x512
        self.lstmcell = nn.LSTMCell(hidden_size, hidden_size)                                # 512, 512

        """ define classifier """
        self.context2out = nn.Linear(context_size, embed_size)                               # 512x512
        self.hidden2tout = nn.Linear(hidden_size, embed_size)                                # 1024x512
        self.dropout = nn.Dropout(p=0.5)

        self.classifier = nn.Linear(embed_size, vocab_size)                                  # 512x8000
        self.init_weight()

        annFile = '/home/gt/D_Data/COCO/annotations_captions/captions_val2014.json'
        self.coco = COCO(annFile)


    def forward(self, images, captions, lengths):

        embeddings = self.embedding(captions)
        packed, batch_sizes = pack_padded_sequence(embeddings, lengths, batch_first=True)
        """ put input data through cnn """
        features = self.encoder(images) # [batch, 512, 14, 14]
        features = features.view(features.size(0), features.size(1), -1).transpose(2, 1) # [batch, 196, 512]
        context_encode = torch.bmm(features, self.image_att_w.unsqueeze(0).expand(features.size(0), self.image_att_w.size(0), self.image_att_w.size(1))) # [batch, 196, 512]

        """ initialize hidden and memory unit"""
        hidden, c = self.init_lstm(features)
        outputs = []

        for t, batch_size in enumerate(batch_sizes):
            embedding = embeddings[:batch_size, t, :]
            context, _ = self.attention_layer(features[:batch_size], context_encode[:batch_size], hidden[:batch_size])

            rnn_input = torch.cat([embedding, context], dim=1)
            hidden, c = self.lstmcell(rnn_input, (hidden[:batch_size], c[:batch_size]))
            output = self.output_layer(context, hidden)

            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        return outputs

    def attention_layer(self, features, context_encode, hidden):
        h_att = F.tanh(context_encode + self.weight_hh(hidden).unsqueeze(1).expand_as(context_encode))
        out_att = torch.bmm(h_att, self.weight_att.unsqueeze(0).expand(h_att.size(0), self.weight_att.size(0), self.weight_att.size(1))).squeeze(2)
        alpha = F.softmax(out_att)
        context = (features * alpha.unsqueeze(2).expand_as(features)).mean(1).squeeze(1)
        return context, alpha

    def output_layer(self, context, hidden, prev=None):
        context = self.context2out(context)
        hidden = self.hidden2tout(hidden)
        out = self.dropout(F.tanh(context+hidden))
        out = self.classifier(out)
        return out


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

    def finetune(self, allow=False):
        for param in self.encoder.parameters():
            param.requires_grad = True if allow else False

    def sample(self, images):
        """"""
        start_word = torch.ones(images.size(0))
        embeddings = self.embedding(Variable(start_word.long()).cuda())
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        embedding_save = []

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

            # # Option 1 : Greedy search
            #predicted_greedy = outputs.max(1)[1]
            #sampled_ids.append(predicted_greedy)
            #embedding = self.embedding(predicted_greedy).squeeze(1)

            # Option 2 : Multinomal search
            predicted_multinomial = outputs.multinomial()
            sampled_ids.append(predicted_multinomial)
            embedding = self.embedding(predicted_multinomial).squeeze(1)
            embedding_save.append(embedding)

            rnn_input = torch.cat([embedding, context], dim=1)

        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)
        embedding_save = torch.cat(embedding_save, 1) # batch, 512*20

        return sampled_ids.squeeze(), embedding_save.view(self.opt.batch_size, 512, 20)

    def gt_sentences(self, captions, lengths):

        r_sentences=[]

        for i in range(len(captions)):

            captions[i]


            r_sentences.append(gt_)

        return r_sentences

    def gumbel_sampler(self, input, tau, temperature):
        noise = torch.rand(input.size())
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        noise = Variable(noise)
        x = (input + noise) / tau + temperature
        x = F.softmax(x.view(input.size(0), -1))
        return x.view_as(input)


class ShowAttendTellModel_D(nn.Module):

    def __init__(self, hidden_size, context_size, vocab_size, embed_size, opt, feature_size=[196, 512]):
        super(ShowAttendTellModel_D, self).__init__()
        self.opt = opt
        self.net_D = nn.Sequential(nn.Linear(self.opt.embed_size, self.opt.embed_size),
                                   nn.ReLU(),
                                   nn.Linear(self.opt.embed_size, 2))

        self.bi_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bias=True, batch_first=True, bidirectional=True)
        self.W_s1 = nn.Linear(256, 1024)
        self.W_s2 = nn.Linear(1, 256)

    def forward(self, input):
            embedding = self.self_attentive_sentence_embedding(input)
            return self.net_D(embedding)

    def self_attentive_sentence_embedding(self, input, I):
        # timesteps, batch, hidden_size // 20, 128, 512x2
        output, hn = self.bi_lstm(input)
        H = torch.transpose(hn, 0, 1) # hn = 20, 128, 512x2 -> 128, 20, 512x2
        H_t = torch.transpose(H, 1, 2) # hn = 128, 20, 512x2 -> 128, 512x2, 20
        temp = F.tanh(self.W_s1(H_t)) # temp = 128, 256, 20
        temp = self.W_s2(temp) # temp = 128, 1,20
        a = F.softmax(temp) # a = 128, 1,20

        embedding = a * H

        return embedding # 128, 256