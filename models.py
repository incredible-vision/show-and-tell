import sys

import torch.nn as nn
import torchvision.models as models

sys.path.append("coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import json
import random
import string
import time
import os
import math
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.resnet.fc.weight.data.normal_(0.0, 0.02)
        self.resnet.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.ss_prob = 0
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embedding = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embedding), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)
        return sampled_ids.squeeze()


class PG_DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(PG_DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.ss_prob = 0
        self.init_weights()

        self.saved_actions = []
        self.rewards = []

        self.annFile = '/home/gt/D_Data/COCO/annotations_captions/captions_train2014.json'
        self.coco = COCO(self.annFile)
        self.valid = self.coco.getImgIds()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths, imgids, vocab):
        """Decode image feature vectors and generates captions."""
        embedding = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embedding), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)

        packed_info = packed[1]
        outputs = self.linear(hiddens[0])

        _, actions = torch.max(outputs, 1)

        Q_s = self.generate_rewards(outputs, captions, imgids, packed_info, vocab)

        return actions, Q_s

    def generate_rewards(self, outputs, captions, imgids, packed_info, vocab):

        captions = captions.unsqueeze(2) # 2, 13, 1
        unpacked_output_s = pad_packed_sequence([outputs, packed_info]) # 13, 2, 10000, [2,2,2... 1,1,1..]
        unpacked_output_s = unpacked_output_s[0].transpose(0, 1) # 2, 13, 10000

        caption_rewards = []
        #    13,10000          13,1     1
        for unpacked_output, caption, imgid in zip(unpacked_output_s, captions, imgids):
            step_rewards = []

            _, temp = torch.max(unpacked_output, 1)
            temp = temp.squeeze(1)

            for t in range(2):#len(unpacked_output)):
                step_reward = 0.0
                pred_fixed = []

                pred_fixed.extend(torch.split(temp, split_size=1, dim=0)[:t+1])

                for k in range(1):
                    pred_k = pred_fixed
                    pred_k.extend(unpacked_output[t+1:].multinomial())

                    sampled_caption = []
                    for word_id in pred_k:
                        word_idx = word_id.data.cpu().numpy()[0]
                        word = vocab.idx2word[word_idx]
                        if word == '<end>': break
                        sampled_caption.append(word)
                    sentence = ' '.join(sampled_caption)
                    pred= {'image_id': imgid, 'caption': sentence}

                    step_reward = step_reward + self.eval_reward_metrics(self.coco, pred)

                step_rewards.append(step_reward/3.0)
            caption_rewards.append(step_rewards)

        return caption_rewards


    def eval_reward_metrics(self, coco, pred):

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


    def sample(self, features, states):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, 20)
        return sampled_ids.squeeze()
