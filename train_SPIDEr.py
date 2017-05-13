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
# from eval import evaluation,
from eval_SPIDEr import evaluationPolicyGradient, language_eval

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

        import sys
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
        sys.path.append("coco-caption")
        self.annFile_train = 'data/MSCOCO/annotations/captions_train2014.json'
        self.coco_train = COCO(self.annFile_train)
        self.valids_train = self.coco_train.getImgIds()
        self.annFile_valid = 'data/MSCOCO/annotations/captions_val2014.json'
        self.coco_valid = COCO(self.annFile_valid)
        self.valids_valid = self.coco_valid.getImgIds()



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

            for epoch in range(1, 1+self.opt.max_epochs):

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
                    self.decoderPolicyGradient.train()

                    # Set mini-batch dataset
                    images   =  Variable(images)
                    captions =  Variable(captions)
                    states   = (Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size)),
                                Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size)))

                    # Set Variables to Support CUDA Computations
                    if self.num_gpu > 0:
                        images   = images.cuda()
                        captions = captions.cuda()
                        states   = [s.cuda() for s in states]

                    # Pack-Padded Sequence for Ground Truth Sentence
                    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                    # Set Gradients of All Models Parameters and Optimizer to Zero
                    self.encoder.zero_grad()
                    self.decoderPolicyGradient.zero_grad()
                    self.optimizer.zero_grad()

                    # Improved Image Captioning via Policy Gradient Optimization of SPIDEr (REINFORCE)
                    # Extract Image Features
                    features = self.encoder(images)

                    # Get Generated Sequence (Scores)
                    #  - Input:  features[128x256], length[128](lengths of each sentence)
                    #  - Output: actions[128x<length>], rewards[128x<length>]
                    maxSequenceLength = max(lengths)
                    outputs = self.decoderPolicyGradient(features, captions, states, maxSequenceLength, lengths, gt=True)

                    # Training the Network using MLE : Calculate loss and Optimize the Network
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    clip_gradient(self.optimizer, self.opt.grad_clip)
                    self.optimizer.step()

                    # Display Information
                    if iter % self.opt.log_step == 0:
                        # Save & Print the Log
                        log_print = '[Loss: MLE] Epoch [%2d/%2d], Step [%4d/%4d], Loss: %2.4f, Perplexity: %6.4f, lr: %1.1e'\
                                    % (epoch, self.opt.max_epochs, iter, self.total_train_iter, loss.data[0], np.exp(loss.data[0]), self.opt.current_lr)
                        print(log_print)
                        savePath = os.path.join(self.opt.expr_dir, self.opt.exp_id + "_MLE_log" + ".txt")
                        with open(savePath, 'a') as f:
                            f.write(log_print)
                            f.write('\n')

                    # Delete Variables
                    del self.decoderPolicyGradient.outputs[:]
                    del self.decoderPolicyGradient.actions[:]
                    del self.decoderPolicyGradient.inputs[:]
                    del self.decoderPolicyGradient.states[:]

                # Make evaluation on validation set, and save model
                val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt, self.coco_valid, self.valids_valid)
                print(lang_stats)

                # Save the Pre-trained Model for Every Epoch
                torch.save(self.encoder.state_dict(),               os.path.join('model/', 'Pre-trained-encoder-epoch%d.pkl'               % (epoch)))
                torch.save(self.decoderPolicyGradient.state_dict(), os.path.join('model/', 'Pre-trained-decoderPolicyGradient-epoch%d.pkl' % (epoch)))

        # Second, Train Baseline, B_phi, using Monte Carlo estimates of Q_theta on a small subset of Dataset.
        for epoch in range(1):
            print('---------------------------------------------------------------------------------------------------')
            print('Second, Train Baseline, B_phi, using Monte Carlo estimates of Q_theta on a small subset of Dataset.')
            print('---------------------------------------------------------------------------------------------------')

        # DEBUG 'model - sample'
        if 0:
            print('---------------------------------------------------------------------------------------------------')
            print('DEBUG:: Evaluate Policy Gradient')
            print('---------------------------------------------------------------------------------------------------')
            # Make evaluation on validation set, and save model
            val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt, self.coco_valid, self.valids_valid)
            print(lang_stats)

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

                    if 0:
                        if len(imgids) < self.opt.batch_size:
                            print('Selected Batch size(%d) is not same as batch size' % len(imgids))
                            print(imgids)
                            continue

                        if 453611 or 521652 in imgids:
                            torch.set_printoptions(edgeitems=100, linewidth=160)
                            print('453611 ERROR')
                            print('imgids')
                            print(imgids)
                            print('captions')
                            print(captions)
                            continue

                    # Set Network model as Training mode
                    self.encoder.train()
                    self.decoderPolicyGradient.train()

                    # Set mini-batch dataset for Policy Gradient
                    imagesVolatile =  Variable(images, volatile=True)
                    statesVolatile = (Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size), volatile=True),
                                      Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size), volatile=True))
                    images   =  Variable(images)
                    captions =  Variable(captions)
                    states   = (Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size)),
                                Variable(torch.zeros(self.opt.num_layers, images.size(0), self.opt.hidden_size)))

                    # Set Variables to Support CUDA Computations
                    if self.num_gpu > 0:
                        imagesVolatile = imagesVolatile.cuda()
                        statesVolatile = [s.cuda() for s in statesVolatile]
                        images = images.cuda()
                        captions = captions.cuda()
                        states = [s.cuda() for s in states]

                    # Pack-Padded Sequence for Ground Truth Sentence
                    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                    # Set Gradients of All Models Parameters and Optimizer to Zero
                    self.encoder.zero_grad()
                    self.decoderPolicyGradient.zero_grad()
                    self.optimizer.zero_grad()

                    # Improved Image Captioning via Policy Gradient Optimization of SPIDEr (REINFORCE)
                    # Extract Image Features
                    features = self.encoder(images)
                    featuresVolatile = self.encoder(imagesVolatile)

                    # Generate Sequence g_1:T ~ Policy(.|Image) with Ground Truth (Training)
                    #  - Input:  features[128x256], length[128](lengths of each sentence)
                    #  - Output: outputs[<length>x10372], actions[128x<length>], rewards[128x<length>]
                    # _, actions, actions_rollouts = self.decoderPolicyGradient(features, captions, states, lengths, self.opt.MC_rollouts, MCRollouts=True)
                    maxSequenceLength = max(lengths)
                    outputs = self.decoderPolicyGradient(features, captions, states, maxSequenceLength, lengths, gt=False)

                    # Calculate loss for Log Display
                    loss = self.criterion(outputs.detach(), targets)

                    # Monte Carlo Rollouts #1 - Every Rollout shares only one LSTM
                    # Monte Carlo Rollouts #2 - Each Rollout has own LSTM
                    actions_rollouts     = self.decoderPolicyGradient.getMonteCarloRollouts(featuresVolatile, statesVolatile, self.opt.MC_rollouts, maxSequenceLength, gt=False)
                    predictions_rollouts = self.decoderPolicyGradient.getSentences(actions_rollouts, imgids, self.vocab)

                    if 0:
                        torch.set_printoptions(edgeitems=100, linewidth=160)
                        for prediction in predictions_rollouts:
                            print(prediction)

                    # Calculate Rewards - Evaluate COCO Metrics
                    rewards_rollouts, lang_stat_rollouts = self.decoderPolicyGradient.getRewardsRollouts(predictions_rollouts, self.opt.MC_rollouts, lengths, maxSequenceLength, self.coco_train, self.valids_train)

                    if 1:
                        for idx, lang_stat in enumerate(lang_stat_rollouts):
                            print('Index: %d, BLEU1: %.4f, BLEU2: %.4f, BLEU3: %.4f, BLEU4: %.4f, CIDER: %.4f, METEOR: %.4f, ROUGE: %.4f'%\
                                  (idx, lang_stat['Bleu_1'], lang_stat['Bleu_2'], lang_stat['Bleu_3'], lang_stat['Bleu_4'], lang_stat['CIDEr'], lang_stat['METEOR'], lang_stat['ROUGE_L']))

                    # Calculate Rewards
                    rewards, rewardsMax, rewardsMin, rewardsAvg = self.decoderPolicyGradient.getRewards(rewards_rollouts, self.opt.MC_rollouts)
                    # Training the Network using REINFORCE ---------------------------------------
                    #  - actions: list [torch.longTensor, torch.longTensor, ..., torch.longTensor]
                    #  -  action: Variable[torch.LongTensor of size 1x1]
                    #  - rewards: [torch.FloatTensor of size 31]
                    #  -       r: float
                    torch.cuda.synchronize()
                    for action, r in zip(self.decoderPolicyGradient.actions[1:], rewards):
                        action.reinforce(r)
                    autograd.backward(self.decoderPolicyGradient.actions[1:], [None for _ in self.decoderPolicyGradient.actions[1:]])
                    clip_gradient(self.optimizer, self.opt.grad_clip)
                    self.optimizer.step()

                    # Display Information :: REINFORCE
                    if iter % self.opt.log_step == 0:
                        # Generate a Log String
                        log_print = '[REINFORCE] Epoch [%2d/%2d], Step [%4d/%4d], Loss: %2.4f, Perplexity: %6.4f, Rewards[min/avg/max]: [%.4f/%.4f/%.4f], Perplexity: [%6.4f/%6.4f/%6.4f], lr: %1.1e' % \
                                    (epoch, self.opt.max_epochs, iter, self.total_train_iter, loss.data[0], np.exp(loss.data[0]), rewardsMin, rewardsAvg, rewardsMax, np.exp(rewardsMin), np.exp(rewardsAvg), np.exp(rewardsMax),
                                     self.opt.current_lr)
                        print(log_print)

                        # Save & Print the Log
                        savePath = os.path.join(self.opt.expr_dir, self.opt.exp_id + "_REINFORCE_log" + ".txt")
                        with open(savePath, 'a') as f:
                            f.write(log_print)
                            f.write('\n')

                        # Save Log - Generated Sentences
                        savePath = os.path.join(self.opt.expr_dir, self.opt.exp_id + "_REINFORCE_GeneratedSentences" + ".txt")
                        with open(savePath, 'a') as f:
                            f.write(log_print)
                            f.write('\n')
                            for prediction in predictions_rollouts:
                                f.write(prediction['caption'])
                                f.write('\n')
                            f.write('\n\n')

                        # Save Log - COCO Metric
                        savePath = os.path.join(self.opt.expr_dir, self.opt.exp_id + "_REINFORCE_COCOMetric" + ".txt")
                        with open(savePath, 'a') as f:
                            f.write(log_print)
                            f.write('\n')
                            for lang_stat in lang_stat_rollouts:
                                log_print_stat = 'BLEU1: %.4f, BLEU2: %.4f, BLEU3: %.4f, BLEU4: %.4f, CIDER: %.4f, METEOR: %.4f, ROUGE: %.4f'%\
                                                 (lang_stat['Bleu_1'], lang_stat['Bleu_2'], lang_stat['Bleu_3'], lang_stat['Bleu_4'], lang_stat['CIDEr'], lang_stat['METEOR'], lang_stat['ROUGE_L'])
                                f.write(log_print_stat)
                                f.write('\n')
                            f.write('\n\n')

                    # Delete Variables
                    del self.decoderPolicyGradient.outputs[:]
                    del self.decoderPolicyGradient.actions[:]
                    del self.decoderPolicyGradient.inputs[:]
                    del self.decoderPolicyGradient.states[:]

                # Make evaluation on validation set, and save model
                val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt, self.coco_valid, self.valids_valid)
                print('Evaluation with Validation Set')
                print(lang_stats)

                # Save the Pre-trained Model for Every Epoch
                torch.save(self.encoder.state_dict(),               os.path.join('model/', 'REINFORCE-encoder-epoch%d.pkl'               % (epoch)))
                torch.save(self.decoderPolicyGradient.state_dict(), os.path.join('model/', 'REINFORCE-decoderPolicyGradient-epoch%d.pkl' % (epoch)))


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