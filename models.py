import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from eval_SPIDEr import language_eval
import os
import time

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
        self.embed = nn.Embedding(vocab_size, embed_size)  # vocab_size: 10372, embed_size: 512
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)  # hidden_size: 512, vocab_size: 10372
        # self.dropOut = nn.Dropout(p=0.5)

        # Baseline Estimator ---------------------------------------
        self.critic_linear = nn.Linear(hidden_size, 1)
        # ----------------------------------------------------------

        self.ss_prob = 0
        self.init_weights()

        # Baseline Estimator ---------------------------------------
        self.values  = []
        # ----------------------------------------------------------
        self.outputs = []
        self.actions = []
        self.inputs  = []
        self.states  = []

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        # Baseline Estimator ---------------------------------------
        self.critic_linear.weight.data = self.normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        # ----------------------------------------------------------

    # Baseline Estimator -------------------------------------------
    def normalized_columns_initializer(self, weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
        return out
    # --------------------------------------------------------------

    def deleteVariables(self):
        del self.values[:]
        del self.outputs[:]
        del self.actions[:]
        del self.inputs[:]
        del self.states[:]

    # At training  time, we always feed in the ground truth(captions) symbol to the RNN decoder.
    def forward(self, features, captions, states, maxlen, lengths, gt=False):
        # Initialize LSTM Input (Image Features)
        inputs = features.unsqueeze(1)
        # Run LSTM with Ground Truth
        if gt:
            for idx in range(0, maxlen):
                # Get Network Output
                hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
                # Baseline Estimator ---------------------------------------
                value  = self.critic_linear(hiddens.squeeze(1).detach())
                # ----------------------------------------------------------
                output = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
                # Dropout
                # output = self.dropOut(output)
                # Get a Stochastic Action (Stochastic Policy)
                # action = self.getStochasticAction(output)
                # Get a Deterministic Action (Deterministic Policy)
                action = self.getDeterministicAction(output)
                # Set LSTM Input with Ground Truth
                inputs = self.embed(captions[:, idx]).unsqueeze(1)
                # Append Results to List Variables
                self.values.append(value)
                self.outputs.append(output)
                self.actions.append(action)
                self.inputs.append(inputs)
                self.states.append(states)
        # Run LSTM with LSTM Output
        else:
            for idx in range(0, maxlen):
                # Get Network Output
                hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
                # Baseline Estimator ---------------------------------------
                value  = self.critic_linear(hiddens.squeeze(1).detach())
                # ----------------------------------------------------------
                output = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
                # Get a Stochastic Action (Stochastic Policy)
                action = self.getStochasticAction(output)
                # Get a Deterministic Action (Deterministic Policy)
                # action = self.getDeterministicAction(output)
                # Set LSTM Input with LSTM Output (Detached!)
                inputs = self.embed(action.detach())
                # Append Results to List Variables
                self.values.append(value)
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

        # DEBUG - Display MC Rollouts
        if 0:
            torch.set_printoptions(edgeitems=maxlen, linewidth=200)
            print(actions_rollouts)

        # Modify Types of Variable
        # actions_rollouts_numpy = actions_rollouts.cpu().numpy()
        return actions_rollouts


    def getMCRolloutSamples(self, inputs, states, actions, t, maxlen):
        for idx in range(t, maxlen-1):
            hiddens, states = self.lstm(inputs, states)
            output = self.linear(hiddens.squeeze(1))
            # Get a Stochastic Action (Stochastic Policy)
            action = self.getStochasticAction(output)
            # Get a Deterministic Action (Deterministic Policy)
            # action = self.getDeterministicAction(output)
            # Set LSTM Input with LSTM Output (Detached!)
            inputs = self.embed(action.detach())
            # Append a Rollout Action to List Variables
            actions.append(action.data)
        return torch.stack(actions, 1).squeeze()

    def getSentences(self, actions_rollouts, imgids, vocab):
        # Initialize List Variables
        predictions = []
        result_sentences = []
        for sentence_ids in actions_rollouts[:, 1:]:  # Without <start>
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
        return predictions

    def getRewardsRollouts(self, predictions_rollouts, batch_size, K, lengths, maxlen, coco_train, valids_train):
        # -------------------------------------------------------------------------------------------------
        # for idx in range(maxlen-1):
        # Evaluate COCO Metrics for Sentences of Each Batch (opt.batch_size = len(lengths))
        # NEval = batch_size * K
        # lang_stat = language_eval(predictions[idx*NEval:(idx+1)*NEval], coco_train, valids_train, NEval)
        # -------------------------------------------------------------------------------------------------
        # Initialize Variables
        rewards_rollouts = []
        lang_stat_rollouts = []
        NEval = batch_size * (maxlen-1) * K
        # Evaluate COCO Metrics
        lang_stat = language_eval(predictions_rollouts, coco_train, valids_train, NEval, batch_size)
        # Modify COCO Metrics
        methods = lang_stat.keys()
        for idx in range(maxlen-1):
            lang_stat_rollout = {}
            for method in methods:
                # tmp1 = lang_stat[method][idx*(batch_size*K):(idx+1)*(batch_size*K)]
                # tmp2 = self.values[idx + 1].data.cpu().numpy().repeat(3)
                # lang_stat_rollout[method] = np.mean(tmp1 - tmp2)
                lang_stat_rollout[method] = np.mean(lang_stat[method][idx*(batch_size*K):(idx+1)*(batch_size*K)])
                # lang_stat_rollout[method] = np.mean(lang_stat[method][idx*(batch_size*K):(idx+1)*(batch_size*K)]) - float(self.values[idx+1].data.cpu().numpy())  # Subtract Baseline
            lang_stat_rollouts.append(lang_stat_rollout)
            # Calculate Reward - BCMR COCO Metric
            BCMR = self.calcBCMR(lang_stat_rollout)
            rewards_rollouts.append(BCMR)
        return rewards_rollouts, lang_stat_rollouts

    def calcBCMR(self, lang_stat):
        BCMR = + 0.5 * lang_stat['Bleu_1'] + 0.5 * lang_stat['Bleu_2'] \
               + 1.0 * lang_stat['Bleu_3'] + 1.0 * lang_stat['Bleu_4'] \
               + 1.0 * lang_stat['CIDEr']  + 5.0 * lang_stat['METEOR'] \
               + 2.0 * lang_stat['ROUGE_L']
        return BCMR

    def getRewards(self, rewards_rollouts, K):
        baseline = []
        for idx in range(1, len(self.values)):
            baseline.append(float(self.values[idx].mean().data.cpu().numpy()))
        rewards = torch.Tensor(np.asarray(rewards_rollouts) - np.asarray(baseline))
        rewardsMax = torch.max(rewards)
        rewardsMin = torch.min(rewards)
        rewardsAvg = torch.mean(rewards)
        return rewards, rewardsMax, rewardsMin, rewardsAvg

    # Get an Action by using Stochastic Policy
    def getStochasticAction(self, output):
        distribution = nn.functional.softmax(output)
        action = distribution.multinomial()
        return action


    # Get an Action by using Deterministic Policy
    def getDeterministicAction(self, output):
        distribution = nn.functional.softmax(output)
        action = distribution.max(1)[1]
        return action


    # Convert Output Variable to Pack Padded Sequence
    def convertOutputVariable(self, maxlen, lengths):
        outputs = torch.cat(self.outputs, 1).view(len(lengths), maxlen, -1)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        return outputs


    # Convert Value Variable to Pack Padded Sequence
    def convertValueVariable(self, maxlen, lengths):
        values = torch.cat(self.values, 1).view(len(lengths), -1)
        values = pack_padded_sequence(values, lengths, batch_first=True)[0]
        return values


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


    # Display & Save Information MLE
    def displaySaveInformationMLE(self, epoch, max_epochs, iter, total_train_iter, loss, current_lr, expr_dir, exp_id):
        savePath = os.path.join(expr_dir, exp_id + "_MLE_log" + ".txt")
        with open(savePath, 'a') as f:
            log_print = '[Loss: MLE] Epoch [%2d/%2d], Step [%4d/%4d], Loss: %2.4f, Perplexity: %6.4f, lr: %1.1e' \
                        % (epoch, max_epochs, iter, total_train_iter, loss.data[0], np.exp(loss.data[0]), current_lr)
            print(log_print)
            f.write(log_print)
            f.write('\n')

    # Display & Save Information COCO Evaluation Metric
    def displaySaveInformationCOCOMetric(self, expr_dir, exp_id, lang_stats, mode):
        if mode == 'MLE':
            savePath = os.path.join(expr_dir, exp_id + "_MLE_log" + ".txt")
        elif mode == 'REINFORCE':
            savePath = os.path.join(expr_dir, exp_id + "_REINFORCE_log" + ".txt")
        with open(savePath, 'a') as f:
            log_print_stat = 'BLEU1: %.4f, BLEU2: %.4f, BLEU3: %.4f, BLEU4: %.4f, CIDER: %.4f, METEOR: %.4f, ROUGE: %.4f' % \
                             (lang_stats['Bleu_1'], lang_stats['Bleu_2'], lang_stats['Bleu_3'], lang_stats['Bleu_4'], lang_stats['CIDEr'], lang_stats['METEOR'], lang_stats['ROUGE_L'])
            print(log_print_stat)
            f.write(log_print_stat)
            f.write('\n\n')

    # Display & Save Information REINFORCE
    def displaySaveInformationREINFORCE(self, epoch, max_epochs, iter, total_train_iter, loss, rewardsMin, rewardsAvg, rewardsMax, current_lr, expr_dir, exp_id, predictions_rollouts, lang_stat_rollouts):
        # Generate a Log String
        log_print = '[REINFORCE] Epoch [%2d/%2d], Step [%4d/%4d], Loss: %2.4f, Perplexity: %6.4f, Rewards[min/avg/max]: [%.4f/%.4f/%.4f], Perplexity: [%6.4f/%6.4f/%6.4f], lr: %1.1e' % \
                    (epoch, max_epochs, iter, total_train_iter,
                     loss.data[0], np.exp(loss.data[0]),
                     rewardsMin, rewardsAvg, rewardsMax,
                     np.exp(rewardsMin), np.exp(rewardsAvg), np.exp(rewardsMax),
                     current_lr)
        print(log_print)

        # Save & Print the Log
        savePath = os.path.join(expr_dir, exp_id + "_REINFORCE_log" + ".txt")
        with open(savePath, 'a') as f:
            f.write(log_print)
            f.write('\n')

        # Save & Print the Log - Generated Sentences
        savePath = os.path.join(expr_dir, exp_id + "_REINFORCE_GeneratedSentences" + ".txt")
        with open(savePath, 'a') as f:
            f.write(log_print)
            f.write('\n')
            for prediction in predictions_rollouts:
                f.write(prediction['caption'])
                f.write('\n')
            f.write('\n\n')

        # Save & Print the Log - COCO Metric
        savePath = os.path.join(expr_dir, exp_id + "_REINFORCE_COCOMetric" + ".txt")
        with open(savePath, 'a') as f:
            f.write(log_print)
            f.write('\n')
            for idx, lang_stat in enumerate(lang_stat_rollouts):
                log_print_stat = 'Rollout index: %3d, BLEU1: %.4f, BLEU2: %.4f, BLEU3: %.4f, BLEU4: %.4f, CIDER: %.4f, METEOR: %.4f, ROUGE: %.4f' % \
                                 (idx, lang_stat['Bleu_1'], lang_stat['Bleu_2'], lang_stat['Bleu_3'], lang_stat['Bleu_4'], lang_stat['CIDEr'], lang_stat['METEOR'], lang_stat['ROUGE_L'])
                print(log_print_stat)
                f.write(log_print_stat)
                f.write('\n')
            f.write('\n\n')

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


