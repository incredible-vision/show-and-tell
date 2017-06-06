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
from torch.autograd import Variable
from Eval_SPIDEr import language_eval


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


#  3.3 Encoder-decoder architecture
#   - Each symbol in the vocabulary is embedded as a 512 dimensional word embedding vector.
#   - Each image is encoded by Inception-V3 as a dense feature vector of dimension 2048 -> 512 dimension with a linear layer and used as the initial state of RNN decoder.
#   - At training  time, we always feed in the ground truth symbol to the RNN decoder.
#   - At inference time, we use just greedy decoding, where the sampled output is fed to the RNN as the next input symbol.
class DecoderPolicyGradient(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderPolicyGradient, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # vocab_size: 10372, embed_size: 256
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)  # hidden_size: 512, vocab_size: 10372
        self.ss_prob = 0
        self.init_weights()

        self.outputs = []
        self.actions = []
        self.inputs  = []
        self.states  = []


    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)


    # At training  time, we always feed in the ground truth(captions) symbol to the RNN decoder.
    def forward(self, features, captions, states, maxlen, lengths, gt=False):
        # Initialize LSTM Input (Image Features)
        inputs = features.unsqueeze(1)
        # Run LSTM with Ground Truth
        if gt:
            for idx in range(0, maxlen):
                # Get Network Output
                hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
                output = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
                # Get a Stochastic Action (Stochastic Policy)
                action = self.getStochasticAction(output)
                # Set LSTM Input with Ground Truth
                inputs = self.embed(captions[:, idx]).unsqueeze(1)
                # Append Results to List Variables
                self.outputs.append(output)
                self.actions.append(action)
                self.inputs.append(inputs)
                self.states.append(states)
        # Run LSTM with LSTM Output
        else:
            for idx in range(0, maxlen):
                # Get Network Output
                hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
                output = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
                # Get a Stochastic Action (Stochastic Policy)
                action = self.getStochasticAction(output)
                # Set LSTM Input with LSTM Output (Detached!)
                inputs = self.embed(action.detach())
                # Append Results to List Variables
                self.outputs.append(output)
                self.actions.append(action)
                self.inputs.append(inputs)
                self.states.append(states)
        # Convert Output Variable to Pack Padded Sequence
        outputs = self.convertOutputVariable(maxlen, lengths)
        return outputs


    # Monte Carlo Rollouts
    def getMonteCarloRollouts(self, features, states, K, maxlen, gt=False):
        # Get MC Rollouts with the LSTM generated with Ground Truth
        if gt:
            # Initialize List Variables
            outputs_detached = []
            actions_detached = []
            actions_rollouts = []
            # Detach Outputs and Actions from the Graph
            for idx in range(maxlen):
                outputs_detached.append(self.outputs[idx].detach())
                actions_detached.append(self.actions[idx].detach())
            # Calculate Number of Columns
            cols = K * (maxlen - 1)  # [(batch_size x K x max_sentence_length)
            # For Initial States,
            for t in range(0, 2):
                # Append Results to List Variables
                actions_rollouts.append(actions_detached[t].data.repeat(cols, 1))
            # For Other States,
            for t in range(2, maxlen):
                # Select Stochastic Actions
                distribution = nn.functional.softmax(outputs_detached[t]).detach()
                # Select Stochastic MC Rollouts & Append Results to List Variable
                actions_rollouts.append(
                    torch.cat((distribution.multinomial(K*(t-1), True).transpose(0, 1).contiguous().view(-1, 1).data,
                               actions_detached[t].data.repeat(cols-K*(t-1), 1)), 0))
            # Modify Types of Variables
            actions_rollouts = torch.stack(actions_rollouts, 1).squeeze()
            if 0:
                torch.set_printoptions(edgeitems=maxlen, linewidth=200)
                print(actions_rollouts)
            # Modify Types of Variable
            actions_rollouts_numpy = actions_rollouts.cpu().numpy()

        # Get MC Rollouts with the LSTM generated without Ground Truth
        else:
            # Initialize LSTM Input (Image Features)
            inputs = features.unsqueeze(1)
            # Initialize List Variables
            outputs_detached = []
            actions_detached = []
            actions_rollouts = []
            # Detach Outputs and Actions from the Graph
            for idx in range(maxlen):
                outputs_detached.append(self.outputs[idx].detach())
                actions_detached.append(self.actions[idx].detach())
            # Initialize List Variables
            actions = []
            # Sampling Actions via Monte Carlo Rollouts
            for t in range(0, maxlen):
                # Get Network Output
                hiddens, states = self.lstm(inputs, states)
                # Get a Stored Stochastic Action (Stochastic Policy)
                action = actions_detached[t]
                # Set LSTM Input with LSTM Output (Detached!)
                inputs = self.embed(action)
                # Append Results to List Variables
                actions.append(actions_detached[t].data)  # Main Actions
                if t >= 1:
                    for k in range(K):
                        inputs_detached = inputs.detach()
                        states_detached = [s.detach() for s in states]
                        actions_copied  = list(actions)
                        actions_rollout = self.getMCRolloutSamples(inputs_detached, states_detached, actions_copied, t, maxlen)
                        actions_rollouts.append(actions_rollout)
            # Modify Types of Variables
            actions_rollouts = torch.stack(actions_rollouts, 0).squeeze().view(-1, maxlen)
            if 0:
                torch.set_printoptions(edgeitems=maxlen, linewidth=200)
                print(actions_rollouts)
            # Modify Types of Variable
            actions_rollouts_numpy = actions_rollouts.cpu().numpy()
        return actions_rollouts_numpy


    def getMCRolloutSamples(self, inputs, states, actions, t, maxlen):
        for idx in range(t, maxlen-1):
            hiddens, states = self.lstm(inputs, states)
            output = self.linear(hiddens.squeeze(1))
            # Get a Stochastic Action (Stochastic Policy)
            action = self.getStochasticAction(output)
            # Set LSTM Input with LSTM Output (Detached!)
            inputs = self.embed(action.detach())
            # Append a Rollout Action to List Variables
            actions.append(action.data)
        return torch.stack(actions, 1).squeeze()

    def getSentences(self, actions_rollouts, imgids, vocab):

        tmp = 453611
        if tmp in imgids:
            print('453611')


        # Initialize List Variables
        predictions = []
        result_sentences = []
        for sentence_ids in actions_rollouts[:, 1:]:
            sampled_caption = []
            for word_id in sentence_ids:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                sampled_caption.append(word)
            sentence = ' '.join(sampled_caption)
            result_sentences.append(sentence)
        for i, sentence in enumerate(result_sentences):
            entry = {'image_id': imgids[i % len(imgids)], 'caption': sentence}

            if isinstance(entry['caption'], list):
                print('453611')

            predictions.append(entry)
        return predictions

    def getRewardsRollouts(self, predictions, K, lengths, maxlen, coco_train, valids_train):
        rewards_rollouts = []
        lang_stat_rollouts = []
        for k in range(K * (maxlen - 1)):
            if 1:
                lang_stat = language_eval(predictions[k * len(lengths):(k + 1) * len(lengths)], coco_train, valids_train)  # Batch-Based
                # lang_stat = language_eval(predictions, coco_train, valids_train)
                BCMR = + 0.5 * lang_stat['Bleu_1'] + 0.5 * lang_stat['Bleu_2'] \
                       + 1.0 * lang_stat['Bleu_3'] + 1.0 * lang_stat['Bleu_4'] \
                       + 1.0 * lang_stat['CIDEr'] + 5.0 * lang_stat['METEOR'] + 2.0 * lang_stat['ROUGE_L']
                lang_stat_rollouts.append(lang_stat)
            else:
                BCMR = 1
            rewards_rollouts.append(BCMR)
        return rewards_rollouts, lang_stat_rollouts

    def getRewards(self, rewards_rollouts, K):
        rewards = []
        for idx in range(len(rewards_rollouts) / K):
            reward = rewards_rollouts[idx * K] + \
                     rewards_rollouts[idx * K + 1] + \
                     rewards_rollouts[idx * K + 2]
            reward = reward / K
            rewards.append(reward)
        rewards = torch.Tensor(rewards)
        rewardsMax = torch.max(rewards)
        rewardsMin = torch.min(rewards)
        rewardsAvg = torch.mean(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)  # baseline
        return rewards, rewardsMax, rewardsMin, rewardsAvg

    # Get an Action by using Stochastic Policy
    def getStochasticAction(self, output):
        distribution = nn.functional.softmax(output)
        # action = torch.multinomial(distribution, 1, True)
        action = distribution.multinomial()
        return action


    # Get an Action by using Deterministic Policy
    def getDeterministicAction(self, output):
        distribution = nn.functional.softmax(output)
        action = distribution.max(1)[1].squeeze()
        return action


    # Convert Output Variable to Pack Padded Sequence
    def convertOutputVariable(self, maxlen, lengths):
        outputs = torch.cat(self.outputs, 1).view(len(lengths), maxlen, -1)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        return outputs


    def display_sentences(self, vocab, actions, imgids):
        actions = actions[:, 1:]  # Rollouts without <start>
        sampled_ids = actions.cpu().data.numpy()
        result_sentences = []
        for sentence_ids in sampled_ids:
            sampled_caption = []
            for word_id in sentence_ids:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                sampled_caption.append(word)
            sentence = ' '.join(sampled_caption)
            result_sentences.append(sentence)
        for i, sentence in enumerate(result_sentences):
            entry = {'image_id': imgids[i % len(imgids)], 'caption': sentence}
            print(entry)


    # RNN Input is 'not' a ground truth symbol(word).
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

