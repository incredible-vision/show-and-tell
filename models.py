import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from eval_SPIDEr import language_eval

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
        self.embed = nn.Embedding(vocab_size, embed_size)  # vocab_size: 10372, embed_size: 256
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)  # hidden_size: 512, vocab_size: 10372
        self.ss_prob = 0
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)  # captions: [128x25], embeddings: [128x25x256]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # [128x26x256] = torch.cat([128x1x256], [128x25x256])
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # ([1635x256], [25]) = pack_padded_sequence([128x26x256], [128]), len(length) = 1635
        hiddens, _ = self.lstm(packed)  # ([1635x512],[25])
        outputs = self.linear(hiddens[0])  # [1635x10372] = self.linear([1635x512]]
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


    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)


    # At training  time, we always feed in the ground truth(captions) symbol to the RNN decoder.
    def forward(self, features, captions, states, lengths):
        # Initialize LSTM Input (Image Features)
        inputs = features.unsqueeze(1)
        # Run LSTM
        for idx in range(0, max(lengths)):
            # Get Network Output
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
            output = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            inputs = self.embed(captions[:, idx]).unsqueeze(1)  # Ground Truth
            # Get a Stochastic Action (Stochastic Policy)
            action = self.getStochasticAction(output)
            # Append Results to List Variables
            self.outputs.append(output)
            self.actions.append(action)
        # Convert Output Variable to Pack Padded Sequence
        outputs = self.convertOutputVariable(lengths)
        return outputs


    # Monte Carlo Rollouts
    def getMonteCarloRollouts(self, K, lengths, imgids, vocab, flag=True):
        if flag:
            # Initialize List Variables
            outputs_detached = []
            actions_detached = []
            actions_rollouts = []
            # Detach Outputs and Actions from the Graph
            for idx in range(max(lengths)):
                outputs_detached.append(self.outputs[idx].detach())
                actions_detached.append(self.actions[idx].detach())
            # Calculate Number of Columns
            cols = K * (max(lengths) - 1)  # [(batch_size x K x max_sentence_length)
            # For Initial States,
            for t in range(0, 2):
                # Append Results to List Variables
                actions_rollouts.append(actions_detached[t].data.repeat(cols, 1))
            # For Other States,
            for t in range(2, max(lengths)):
                # Select Stochastic Actions
                distribution = nn.functional.softmax(outputs_detached[t]).detach()
                # Select Stochastic MC Rollouts & Append Results to List Variable
                actions_rollouts.append(
                    torch.cat((distribution.multinomial(K*(t-1), True).transpose(0, 1).contiguous().view(-1, 1).data,
                               actions_detached[t].data.repeat(cols-K*(t-1), 1)), 0))
            # Modify Types of Variables
            actions_rollouts = torch.stack(actions_rollouts, 1).squeeze()
            if 0:
                print('---------------------')
                torch.set_printoptions(edgeitems=100, linewidth=160)
                print('actions')
                print(self.actions)
                print('actions_rollouts')
                print(actions_rollouts)
                print('---------------------')
            # Initialize List Variables
            predictions = []
            result_sentences = []
            # Modify Types of Variable
            actions_rollouts_numpy = actions_rollouts.cpu().numpy()
            for sentence_ids in actions_rollouts_numpy[:, 1:]:
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
                predictions.append(entry)

        else:
            print('---')

        return predictions


    # Get an Action by using Stochastic Policy
    def getStochasticAction(self, output):
        distribution = nn.functional.softmax(output)
        action = torch.multinomial(distribution, 1, True)
        return action


    # Get an Action by using Deterministic Policy
    def getDeterministicAction(self, output):
        distribution = nn.functional.softmax(output)
        action = distribution.max(1)[1].squeeze()
        return action


    # Convert Output Variable to Pack Padded Sequence
    def convertOutputVariable(self, lengths):
        outputs = torch.cat(self.outputs, 1).view(len(lengths), max(lengths), -1)
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

