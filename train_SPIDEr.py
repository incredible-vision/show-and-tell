import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import pickle
import json
from data_loader import get_loader
from utils import Vocabulary
from models import EncoderCNN, DecoderRNN, DecoderPolicyGradient
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms
from eval import evaluation, evaluationPolicyGradient
from eval_SPIDEr import language_eval


# Improved Image Captioning via Policy Gradient Optimization of SPIDEr (REINFORCE)
class Trainer(object):
    def __init__(self, opt, trainloader, validloader):

        print('--------------------------------------------------------------------------------')
        print('Improved Image Captioning via Policy Gradient Optimization of SPIDEr (REINFORCE)')
        print('--------------------------------------------------------------------------------')

        self.opt = opt

        self.total_train_iter = len(trainloader)
        self.total_valid_iter = len(validloader)

        self.trainloader = trainloader
        self.validloader = validloader

        self.num_gpu = opt.num_gpu
        self.load_model_path = opt.load_model_path
        self.load_optimizer_path = opt.load_optim_path

        with open(opt.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        self.encoder = EncoderCNN(opt.embed_size)
        # self.decoder = DecoderRNN(opt.embed_size, opt.hidden_size, len(self.vocab), opt.num_layers)
        self.decoderPolicyGradient = DecoderPolicyGradient(opt.embed_size, opt.hidden_size, len(self.vocab), opt.num_layers)

        if self.num_gpu == 1:
            self.encoder.cuda()
            # self.decoder.cuda()
            self.decoderPolicyGradient.cuda()

        elif self.num_gpu > 1:
            self.encoder = nn.DataParallel(self.encoder.cuda(), device_ids=range(self.num_gpu))
            # self.decoder = nn.DataParallel(self.decoder.cuda(), device_ids=range(self.num_gpu))
            self.decoderPolicyGradient = nn.DataParallel(self.decoderPolicyGradient.cuda(), device_ids=range(self.num_gpu))

        if self.load_model_path:
            self.load_model()

        if self.load_optimizer_path:
            self.load_optimizer()

        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(list(self.encoder.resnet.fc.parameters())+list(self.decoder.parameters()), lr=opt.learning_rate)
        self.optimizer = optim.Adam(list(self.encoder.resnet.fc.parameters()) + list(self.decoderPolicyGradient.parameters()), lr=opt.learning_rate)

    def load_model(self):
        """"""

    def load_optimizer(self):
        """"""

    def train(self):

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

        def set_lr(optimizer, lr):
            for group in optimizer.param_groups:
                group['lr'] = lr

        # First, Pre-Train the Model using Maximum Likelihood Estimation on Dataset.
        if 1:
            # Load the Pre-Trained Models
            print('----------------------------------')
            print('First, Load the Pre-Trained Models')
            print('----------------------------------')
            self.encoder.load_state_dict(torch.load('model/Pre-trained-encoder-epoch20.pkl'))
            self.decoderPolicyGradient.load_state_dict(torch.load('model/Pre-trained-decoderPolicyGradient-epoch20.pkl'))

        else:
            print('--------------------------------------------------------------------------')
            print('First, Pre-Train the Model using Maximum Likelihood Estimation on Dataset.')
            print('--------------------------------------------------------------------------')
            for epoch in range(self.opt.max_epochs):

                # Update Learning Rate
                if epoch > self.opt.learning_rate_decay_start and self.opt.learning_rate_decay_start >= 1:
                    fraction = (epoch - self.opt.learning_rate_decay_start) // self.opt.learning_rate_decay_every
                    decay_factor = self.opt.learning_rate_decay_rate ** fraction
                    self.opt.current_lr = self.opt.learning_rate * decay_factor
                    set_lr(self.optimizer, self.opt.current_lr)
                else:
                    self.opt.current_lr = self.opt.learning_rate

                # Assign the scheduled sampling prob
                if 0:
                    if epoch > self.opt.scheduled_sampling_start and self.opt.scheduled_sampling_start >= 0:
                        fraction = (epoch - self.opt.scheduled_sampling_start) // self.opt.scheduled_sampling_increase_every
                        self.opt.ss_prob = min(self.opt.scheduled_sampling_increase_prob * fraction, self.opt.scheduled_sampling_max_prob)
                        self.decoder.ss_prob = self.opt.ss_prob

                # For each Iteration,
                for iter, (images, captions, lengths, imgids) in enumerate(self.trainloader):
                    # Set Network model as Training mode
                    self.encoder.train()
                    # self.decoder.train()
                    self.decoderPolicyGradient.train()

                    # Set Timer
                    torch.cuda.synchronize()
                    start = time.time()

                    # Set mini-batch dataset
                    images   = Variable(images)
                    captions = Variable(captions)
                    states   = (Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size)),
                                Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size)))


                    if self.num_gpu > 0:
                        images   = images.cuda()
                        captions = captions.cuda()
                        states   = [s.cuda() for s in states]

                    # Pack-Padded Sequence for Ground Truth Sentence
                    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                    # Forward, Backward and Optimize
                    self.encoder.zero_grad()
                    # self.decoder.zero_grad()
                    self.decoderPolicyGradient.zero_grad()

                    # Improved Image Captioning via Policy Gradient Optimization of SPIDEr (REINFORCE)
                    # Extract Image Features
                    #  - Input:  images[128x3x224x224]
                    #  - Output: features[128x256]
                    features = self.encoder(images)

                    # Get Generated Sequence (Scores)
                    #  - Input:  features[128x256], length[128](lengths of each sentence)
                    #  - Output: actions[128x<length>], rewards[128x<length>]
                    outputs, _, _ = self.decoderPolicyGradient(features, captions, states, lengths, self.opt.MC_rollouts, MCRollouts=False)

                    # Training the Network using MLE
                    # Calculate loss and Optimize the Network
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    clip_gradient(self.optimizer, self.opt.grad_clip)
                    self.optimizer.step()

                    # Get Timer
                    torch.cuda.synchronize()
                    end = time.time()

                    # Display Information
                    if iter % self.opt.log_step == 0:
                        print('[Loss: MLE] Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                              % (epoch, self.opt.max_epochs, iter, self.total_train_iter, loss.data[0], np.exp(loss.data[0])))

                # Make evaluation on validation set, and save model
                val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt)

                # Save the Pre-trained Model for Every Epoch
                torch.save(self.encoder.state_dict(),               os.path.join('model/', 'Pre-trained-encoder-epoch%d.pkl'               % (epoch + 1)))
                torch.save(self.decoderPolicyGradient.state_dict(), os.path.join('model/', 'Pre-trained-decoderPolicyGradient-epoch%d.pkl' % (epoch + 1)))

        # Second, Train Baseline, B_phi, using Monte Carlo estimates of Q_theta on a small subset of Dataset.
        for epoch in range(1):
            print('---------------------------------------------------------------------------------------------------')
            print('Second, Train Baseline, B_phi, using Monte Carlo estimates of Q_theta on a small subset of Dataset.')
            print('---------------------------------------------------------------------------------------------------')

        # DEBUG 'model - sample'
        if 0:
            print('DEBUG:: Evaluate Policy Gradient')
            # Make evaluation on validation set, and save model
            val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt)

        # Third, Train the Model using REINFORCE with Monte Carlo Rollouts
        if 1:
            print('-----------------------------------------------------------------')
            print('Third, Train the Model using REINFORCE with Monte Carlo Rollouts.')
            print('-----------------------------------------------------------------')
            for epoch in range(1, 1 + self.opt.max_epochs_REINFORCE):

                # Update Learning Rate
                if epoch > self.opt.learning_rate_decay_start and self.opt.learning_rate_decay_start >= 1:
                    fraction = (epoch - self.opt.learning_rate_decay_start) // self.opt.learning_rate_decay_every
                    decay_factor = self.opt.learning_rate_decay_rate ** fraction
                    self.opt.current_lr = self.opt.learning_rate * decay_factor
                    set_lr(self.optimizer, self.opt.current_lr)
                else:
                    self.opt.current_lr = self.opt.learning_rate

                # For each Iteration,
                for iter, (images, captions, lengths, imgids) in enumerate(self.trainloader):

                    # Set Network model as Training mode
                    self.encoder.train()
                    self.decoderPolicyGradient.train()

                    # Set Timer
                    # torch.cuda.synchronize()
                    # start = time.time()

                    # Set mini-batch dataset for Policy Gradient
                    images   = Variable(images)
                    captions = Variable(captions)
                    states   = (Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size)),
                                Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size)))

                    # Set GPU
                    if self.num_gpu > 0:
                        # Set mini-batch dataset for Policy Gradient
                        images = images.cuda()
                        captions = captions.cuda()
                        states = [s.cuda() for s in states]

                    # Pack-Padded Sequence for Ground Truth Sentence
                    # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                    # Forward, Backward and Optimize
                    self.encoder.zero_grad()
                    self.decoderPolicyGradient.zero_grad()

                    # Improved Image Captioning via Policy Gradient Optimization of SPIDEr (REINFORCE)
                    # Extract Image Features
                    #  - Input:  images[128x3x224x224]
                    #  - Output: features[128x256]
                    features          = self.encoder(images)

                    # Generate Sequence g_1:T ~ Policy(.|Image) with Ground Truth (Training)
                    #  - Input:  features[128x256], length[128](lengths of each sentence)
                    #  - Output: outputs[<length>x10372], actions[128x<length>], rewards[128x<length>]
                    # _, actions, actions_rollouts = self.decoderPolicyGradient(features, captions, states, lengths, self.opt.MC_rollouts, MCRollouts=True)
                    _ = self.decoderPolicyGradient(features, captions, states, lengths, self.opt.MC_rollouts, MCRollouts=True)
                    '''
                    # Convert actions_rollouts(word indexes) to sentences
                    predictions = []
                    # sampled_ids = actions_rollouts[:, 1:].cpu().data.numpy()  # Rollouts without <start>
                    # sampled_ids = actions_rollouts.cpu().data.numpy()
                    sampled_ids = actions_rollouts
                    sampled_ids = sampled_ids[:, 1:]
                    result_sentences = []
                    for sentence_ids in sampled_ids:
                        sampled_caption = []
                        for word_id in sentence_ids:
                            word = self.vocab.idx2word[word_id]
                            if word == '<end>':
                                break
                            sampled_caption.append(word)
                        sentence = ' '.join(sampled_caption)
                        result_sentences.append(sentence)
                    for i, sentence in enumerate(result_sentences):
                        entry = {'image_id': imgids[i % len(imgids)], 'caption': sentence}
                        predictions.append(entry)

                    '''

                    if 0:
                        torch.set_printoptions(edgeitems=100, linewidth=160)

                    # Calculate Rewards - Evaluate COCO Metrics
                    rewards = []
                    rewards_rollouts = []
                    lang_stat_rollouts = []
                    for k in range(self.opt.MC_rollouts*(max(lengths)-1)):
                        if 0:
                            lang_stat = language_eval(predictions[k * len(lengths):(k + 1) * len(lengths)])  # Batch-Based
                            BCMR = + 0.5 * lang_stat['Bleu_1'] + 0.5 * lang_stat['Bleu_2'] \
                                   + 1.0 * lang_stat['Bleu_3'] + 1.0 * lang_stat['Bleu_4'] \
                                   + 1.0 * lang_stat['CIDEr']  + 5.0 * lang_stat['METEOR'] + 2.0 * lang_stat['ROUGE_L']
                            lang_stat_rollouts.append(lang_stat)
                        BCMR = 1
                        rewards_rollouts.append(BCMR)

                    for idx in range(len(rewards_rollouts)/self.opt.MC_rollouts):
                        reward = rewards_rollouts[idx*self.opt.MC_rollouts] + \
                              rewards_rollouts[idx*self.opt.MC_rollouts+1] + \
                              rewards_rollouts[idx*self.opt.MC_rollouts+2]
                        reward = reward / self.opt.MC_rollouts
                        rewards.append(reward)
                    rewards = torch.Tensor(rewards)
                    rewards_max = torch.max(rewards)
                    rewards_min = torch.min(rewards)
                    rewards_avg = torch.mean(rewards)
                    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)  # baseline

                    # Training the Network using REINFORCE ---------------------------------------
                    #  - actions: list [torch.longTensor, torch.longTensor, ..., torch.longTensor]
                    #  -  action: Variable[torch.LongTensor of size 1x1]
                    #  - rewards: [torch.FloatTensor of size 31]
                    #  -       r: float
                    for action, r in zip(self.decoderPolicyGradient.actions[1:], rewards):
                        action.reinforce(r)
                    self.optimizer.zero_grad()
                    autograd.backward(self.decoderPolicyGradient.actions[1:], [None for _ in self.decoderPolicyGradient.actions[1:]])
                    clip_gradient(self.optimizer, self.opt.grad_clip)
                    self.optimizer.step()

                    # Get Timer
                    # torch.cuda.synchronize()
                    # end = time.time()

                    del images
                    del captions
                    del states
                    del features
                    del self.decoderPolicyGradient.actions[:]
                    self.decoderPolicyGradient.actions_rollouts = []
                    del rewards
                    del rewards_rollouts[:]

                    # Display Information :: REINFORCE
                    if iter % self.opt.log_step == 0:
                        print('[REINFORCE] Epoch [%d/%d], Step [%d/%d], Rewards[min/avg/max]: [%.4f/%.4f/%.4f], Perplexity: [%5.4f/%5.4f/%5.4f]'
                              % (epoch, self.opt.max_epochs, iter, self.total_train_iter, rewards_min, rewards_avg, rewards_max, np.exp(rewards_min), np.exp(rewards_avg), np.exp(rewards_max)))

                # Make evaluation on validation set, and save model
                val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt)

                # Save the Pre-trained Model for Every Epoch
                torch.save(self.encoder.state_dict(),               os.path.join('model/', 'PolicyGradient-encoder-epoch%d.pkl'               % (epoch + 1)))
                torch.save(self.decoderPolicyGradient.state_dict(), os.path.join('model/', 'PolicyGradient-decoderPolicyGradient-epoch%d.pkl' % (epoch + 1)))


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    # parser.add_argument('--root_dir', type=str, default='/home/myunggi/Research/show-and-tell', help="root directory of the project")
    parser.add_argument('--root_dir', type=str, default='/home/dehlix/Projects/Captioning/show-and-tell', help="root directory of the project")
    parser.add_argument('--data_json', type=str, default='data/data.json', help='input data list which includes captions and image information')
    parser.add_argument('--crop_size', type=int, default=224, help='image crop size')

    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')


    parser.add_argument('--log_step', type=int, default=10,help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))
    main(args)