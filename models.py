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

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)


    def MonteCarloRollouts(self, inputs, hiddens, states, outputs, t, maxlen, K):
        # Initialize Output Variables
        rollouts = outputs.repeat(K, 1)
        for idx in range(t, maxlen):
            # Get Network Output Distribution
            distribution = nn.functional.softmax(outputs)
            # Stochastic Sampling for MC Rollouts
            prediction = torch.cat(([distribution.multinomial() for _ in range(K)]))
            # Deterministic Sampling for MC Rollouts
            # prediction = torch.cat(([distribution.max(1)[1] for _ in range(K)]))
            rollouts.append(prediction)
        return rollouts

    # At training  time, we always feed in the ground truth(captions) symbol to the RNN decoder.
    def forward(self, features, captions, states, lengths, K, MCRollouts=False):
        # Initialize Output Variables
        outputs = []
        actions = []
        actions_rollouts = []  # actions_rollouts = Variable(torch.zeros(len(lengths)*K*max(lengths), max(lengths))).cuda().long()

        # Initialize LSTM Input (Image Features)
        inputs = features.unsqueeze(1)
        cols = K * (max(lengths)-1)  # [(batch_size x K x max_sentence_length)

        # Monte Carlo Rollouts
        if MCRollouts:
            # For Initial State,
            for t in range(0, 2):
                # Initialize LSTM Network
                hiddens, states = self.lstm(inputs, states)         # (batch_size, 1, hidden_size)
                output = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
                inputs = self.embed(captions[:, t]).unsqueeze(1)    # Ground Truth
                outputs.append(output)
                # Select Stochastic Actions
                distribution = nn.functional.softmax(output)
                action = torch.multinomial(distribution, 1, True)  # {Variable} [torch.LongTensor of size 1x1]
                # Append Results to List Variables
                actions.append(action)
                actions_rollouts.append(action.repeat(cols, 1))
            # For Every State,
            for t in range(2, max(lengths)):
                # Get LSTM Network Output
                hiddens, states = self.lstm(inputs, states)         # (batch_size, 1, hidden_size)
                output = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
                inputs = self.embed(captions[:, t]).unsqueeze(1)    # Ground Truth
                outputs.append(output)
                # Select Stochastic Actions
                distribution = nn.functional.softmax(output)
                action = torch.multinomial(distribution, 1, True)  # {Variable} [torch.LongTensor of size 1x1]
                # Select Stochastic MC Rollouts & Append Results to List Variable
                actions.append(action)
                self.eval()
                actions_rollouts.append(
                    torch.cat((torch.multinomial(distribution.repeat(K*(t-1), 1), 1, True),
                               action.repeat(cols-K*(t-1), 1)), 0))
                self.train()

            # Modify Types of Variables
            actions_rollouts = torch.stack(actions_rollouts, 1).squeeze()
            # Modify Types of Variables
            outputs = torch.cat(outputs, 1).view(len(lengths), max(lengths), -1)
            # Modify Types of Variables into Pack-padded Sequence
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
            if 0:
                print('---------------------')
                torch.set_printoptions(edgeitems=100, linewidth=160)
                print('actions')
                print(actions)
                print('actions_rollouts')
                print(actions_rollouts)
                print('---------------------')

        # Vanilla Training
        else:
            for idx in range(0, max(lengths)):
                # Get Network Output
                hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
                output = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
                inputs = self.embed(captions[:, idx]).unsqueeze(1)  # Ground Truth
                outputs.append(output)
            # Modify Types of Variables
            outputs = torch.cat(outputs, 1).view(len(lengths), max(lengths), -1)
            # Modify Types of Variables into Pack-padded Sequence
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]

        return outputs, actions, actions_rollouts


    # Get Pack-padded sequence batch sizes.
    def get_batch_sizes(self, lengths):
        tmp = np.zeros([len(lengths), max(lengths)], dtype='int32')
        for idx in range(len(lengths)):
            tmp[idx, 0:lengths[idx]] += 1
        return np.sum(tmp, 0).tolist()


    # Modify a type of the 'actions' variable to user-modified Pad-packed sequences
    def modify_typeof_actions(self, actions, batch_sizes):
        var_data = actions
        max_batch_size = batch_sizes[0]
        output = var_data.data.new(len(batch_sizes), max_batch_size, *var_data.size()[1:]).zero_()
        output = Variable(output)
        lengths = []
        data_offset = 0
        prev_batch_size = batch_sizes[0]
        for i, batch_size in enumerate(batch_sizes):
            output[i, :batch_size] = var_data[data_offset:data_offset + batch_size]
            data_offset += batch_size
            dec = prev_batch_size - batch_size
            if dec > 0:
                lengths.extend((i,) * dec)
            prev_batch_size = batch_size
        lengths.extend((i + 1,) * batch_size)
        lengths.reverse()
        output = output.transpose(0, 1)
        # output += (output == 0).long() * 2  # Modify 0 to <end>
        return output


    # Get Actions by using Stochastic Policy
    def get_stochastic_actions(self, outputs, batch_sizes):
        distribution = nn.functional.softmax(outputs)
        distribution = self.modify_typeof_actions(distribution, batch_sizes)
        actions = distribution.multinomial()
        return actions

        '''
        distribution = nn.functional.softmax(outputs)
        actions = distribution.multinomial().squeeze()
        return self.modify_typeof_actions(actions, batch_sizes)
        '''


    # Get Actions by using Deterministic Policy
    def get_deterministic_actions(self, outputs, batch_sizes):
        distribution = nn.functional.softmax(outputs)
        actions = distribution.max(1)[1].squeeze()
        return self.modify_typeof_actions(actions, batch_sizes)


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


    # Monte Carlo Rollouts
    def MC_Rollouts(self, actions, features, captions, states_init, t, lengths, K, vocab, imgids):
        # Initialize Dump Variable for Selected Actions in MC Rollouts
        actions_rollouts = []
        # Initialize variables
        predictions = []
        # Initialize LSTM Input
        inputs = features.unsqueeze(1)
        states = states_init
        # Sample actions by using Monte Carlo Rollouts
        for idx in range(0, max(lengths)-1):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            # Initialize
            if idx == 0:
                actions_rollouts = actions[:, idx].unsqueeze(1)
                actions_rollouts = actions_rollouts.repeat(K, 1)
            # For Given State,
            elif idx <= t:
                prediction = actions[:, idx].unsqueeze(1)
                prediction = prediction.repeat(K, 1)
                actions_rollouts = torch.cat((actions_rollouts, prediction), 1).squeeze()
            # MC Rollouts
            else:
                distribution = nn.functional.softmax(outputs)
                # Stochastic Sampling for MC Rollouts
                prediction = torch.cat(([distribution.multinomial() for _ in range(K)]))
                # Deterministic Sampling for MC Rollouts
                # prediction = torch.cat(([distribution.max(1)[1] for _ in range(K)]))
                actions_rollouts = torch.cat((actions_rollouts, prediction), 1).squeeze()
            inputs = self.embed(captions[:, idx]).unsqueeze(1)  # Ground Truth
        actions_rollouts = torch.cat((actions_rollouts, actions[:, idx+1].unsqueeze(1).repeat(K, 1)), 1)

        # Calculate Reward
        actions_rollouts_without_start = actions_rollouts[:, 1:]  # Rollouts without <start>
        sampled_ids = actions_rollouts_without_start.cpu().data.numpy()
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
            predictions.append(entry)

        if 0:
            print('-----------------')
            print('actions_rollouts')
            print(actions_rollouts)
            print('-----------------')
            print('-----------------')
            print('predictions')
            print(predictions)
            print('-----------------')

        # Evaluate COCO Metrics
        rewards = []
        lang_stats = []
        for k in range(K):
            lang_stat = language_eval(predictions[k*len(lengths):(k+1)*len(lengths)])
            BCMR = 0.5 * lang_stat['Bleu_1'] + 0.5 * lang_stat['Bleu_2'] \
                 + 1.0 * lang_stat['Bleu_3'] + 1.0 * lang_stat['Bleu_4'] \
                 + 1.0 * lang_stat['CIDEr'] + 5.0 * lang_stat['METEOR'] + 2.0 * lang_stat['ROUGE_L']
            lang_stats.append(lang_stat)
            rewards.append(BCMR)


        # Calculate Weighted Combination of COCO Metrics (BCMR)
        if 0:
            print('-------------------------')
            print('rewards')
            print(rewards)
            print('-------------------------')

        return np.mean(np.array(rewards)), actions_rollouts


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

    # At inference time, we use just greedy decoding, where the sampled output is fed to the RNN as the next input symbol.
    def inference(self, features, lengths):
        """Samples captions for given image features (Greedy search)."""
        inputs = features.unsqueeze(1)
        # For each action(word),
        for i in range(20):  # maximum sampling length
            hiddens, state = self.lstm(inputs, state)  # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)

            # Select a stochastic action(word)
            probs = nn.Softmax(outputs)
            action = probs.multinomial()
            actions.append(action)

            # Evaluate Reward via COCO Metrics
            reward = 0
            for k in range(3):
                # Monte Carlo Rollouts
                #  - Input:  actions[128x(i+1)](Sequence of words chosen so far)
                #  - Output: rollouts[128x<random>](k-th rollout sequence)
                rollout = monteCarloRollout(actions)
                reward = reward + evaluateMetrics(rollout)

            # Add reward to the end of the list
            rewards.append(reward/3.0)

            # If EOF, break
            if action == '<end>':
                break

            # Set a next state
            state = self.embed(action)

        return actions, rewards




    def evaluateMetrics(self, rollouts):
        reward = []
        return reward