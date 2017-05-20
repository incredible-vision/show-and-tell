import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import pickle
import json
from data_loader import get_loader
from utils import Vocabulary, clip_gradient, set_lr
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
    def __init__(self, opt, trainloader, validloader, load_model_path, load_model_flag=False):

        # Initialize Parameters
        self.opt = opt
        self.num_gpu = opt.num_gpu
        self.load_model_flag = load_model_flag
        self.load_model_path = load_model_path
        self.load_optimizer_flag = opt.load_optim_flag
        self.load_optimizer_path = opt.load_optim_path

        # Initialize Train & Valid Loaders
        self.total_train_iter = len(trainloader)  # Number of Iterations (Train)
        self.total_valid_iter = len(validloader)  # Number of Iterations (Valid)
        self.trainloader = trainloader
        self.validloader = validloader

        # Load a vocab
        with open(opt.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Initialize Network Models
        self.encoder = EncoderCNN(opt.embed_size)
        self.decoderPolicyGradient = DecoderPolicyGradient(opt.embed_size, opt.hidden_size, len(self.vocab), opt.num_layers)
        if self.load_model_flag:
            self.load_model(self.load_model_path)

        # Initialize CUDA
        if self.num_gpu == 1:
            self.encoder.cuda()
            self.decoderPolicyGradient.cuda()
        elif self.num_gpu > 1:
            self.encoder = nn.DataParallel(self.encoder.cuda(), device_ids=range(self.num_gpu))
            self.decoderPolicyGradient = nn.DataParallel(self.decoderPolicyGradient.cuda(), device_ids=range(self.num_gpu))

        # Initialize Optimizers
        self.optimizer        = optim.Adam(list(self.encoder.resnet.fc.parameters()) +
                                           list(self.decoderPolicyGradient.parameters())[0:7], lr=opt.learning_rate)
        self.optimizer_critic = optim.Adam(list(self.decoderPolicyGradient.parameters())[8:9], lr=opt.learning_rate)
        if self.load_optimizer_flag:
            self.load_optimizer(self.load_optimizer_path)

        # Initialize Criterions (Loss)
        self.criterion        = nn.CrossEntropyLoss()
        self.criterion_critic = nn.MSELoss()

        # Initialize COCO
        import sys
        from pycocotools.coco import COCO
        sys.path.append("coco-caption")
        self.annFile_train = 'data/MSCOCO/annotations/captions_train2014.json'
        self.coco_train    = COCO(self.annFile_train)
        self.valid_train   = self.coco_train.getImgIds()
        self.annFile_valid = 'data/MSCOCO/annotations/captions_val2014.json'
        self.coco_valid    = COCO(self.annFile_valid)
        self.valid_valid   = self.coco_valid.getImgIds()


    # Load Pre-trained Network Models
    def load_model(self, loc):
        print('Loading the Pre-Trained Models.')
        # self.encoder.load_state_dict(torch.load('model/Pre-trained-encoder-epoch31.pkl'))
        # self.decoderPolicyGradient.load_state_dict(torch.load('model/Pre-trained-decoderPolicyGradient-epoch31.pkl'))
        self.encoder.load_state_dict(torch.load(loc[0]))
        self.decoderPolicyGradient.load_state_dict(torch.load(loc[1]))


    # Load Saved Optimizer
    def load_optimizer(self):
        print('Loading the Saved Optimizers.')


    # Pre-Train the Show and Tell Model using Maximum Likelihood Estimation on Training Dataset.
    def trainMLE(self):
        # First, Pre-Train the Show and Tell Model using Maximum Likelihood Estimation on Training Dataset.
        if 1:
            # For each Epoch,
            for epoch in range(1, 1+self.opt.max_epochs):
                # Update Learning Rate
                if epoch > self.opt.learning_rate_decay_start and self.opt.learning_rate_decay_start >= 1:
                    fraction = (epoch - self.opt.learning_rate_decay_start) // self.opt.learning_rate_decay_every
                    decay_factor = self.opt.learning_rate_decay_rate ** fraction
                    self.opt.current_lr = self.opt.learning_rate * decay_factor
                    set_lr(self.optimizer, self.opt.current_lr)
                else:
                    self.opt.current_lr = self.opt.learning_rate

                # Set Network model as Training mode
                self.encoder.train()
                self.decoderPolicyGradient.train()

                # For each Iteration,
                for iter, (images, captions, lengths, imgids) in enumerate(self.trainloader):

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
                    self.optimizer_critic.zero_grad()

                    # Extract Image Features
                    features = self.encoder(images)

                    # Get Generated Sequence (Scores)
                    maxSequenceLength = max(lengths)
                    outputs = self.decoderPolicyGradient(features, captions, states, maxSequenceLength, lengths, gt=True)

                    # Training the Network using MLE : Calculate loss and Optimize the Network
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    clip_gradient(self.optimizer, self.opt.grad_clip)
                    self.optimizer.step()

                    # Display Information
                    if iter % self.opt.log_step == 0:
                        self.decoderPolicyGradient.displaySaveInformationMLE(epoch, self.opt.max_epochs, iter, self.total_train_iter, loss, self.opt.current_lr, self.opt.expr_dir, self.opt.exp_id)

                    # Delete Variables
                    self.decoderPolicyGradient.deleteVariables()

                # Make evaluation on validation set, and save model
                val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt, self.coco_valid, self.valid_valid, self.opt.batch_size)
                # Display & Save Information :: MLE Show and Tell
                self.decoderPolicyGradient.displaySaveInformationCOCOMetric(self.opt.expr_dir, self.opt.exp_id, lang_stats, mode='MLE')
                # Save the Pre-trained Model for Every Epoch
                torch.save(self.encoder.state_dict(),               os.path.join('model/', self.opt.exp_id + '_Pre-trained-encoder-epoch%d.pkl'               % (epoch)))
                torch.save(self.decoderPolicyGradient.state_dict(), os.path.join('model/', self.opt.exp_id + '_Pre-trained-decoderPolicyGradient-epoch%d.pkl' % (epoch)))



    # DEBUG ::: Evaluate COCO Metrics on Validation Dataset.
    def validation(self):
        # Make evaluation on validation set, and save model
        val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt, self.coco_valid, self.valid_valid, self.opt.batch_size)
        log_print_stat = 'BLEU1: %.4f, BLEU2: %.4f, BLEU3: %.4f, BLEU4: %.4f, CIDER: %.4f, METEOR: %.4f, ROUGE: %.4f' % \
                         (lang_stats['Bleu_1'], lang_stats['Bleu_2'], lang_stats['Bleu_3'], lang_stats['Bleu_4'], lang_stats['CIDEr'], lang_stats['METEOR'], lang_stats['ROUGE_L'])
        print(log_print_stat)


    # Train the SPIDEr Model using Policy Gradient (REINFORCE) with Monte Carlo Rollouts on Training Dataset
    def trainREINFORCE(self):
        # For each Epoch,
        for epoch in range(1, 1 + self.opt.max_epochs_REINFORCE):

            # Update Learning Rate
            if epoch > self.opt.learning_rate_decay_start and self.opt.learning_rate_decay_start >= 1:
                fraction = (epoch - self.opt.learning_rate_decay_start) // self.opt.learning_rate_decay_every
                decay_factor = self.opt.learning_rate_decay_rate ** fraction
                self.opt.current_lr = self.opt.learning_rate_REINFORCE * decay_factor
                set_lr(self.optimizer, self.opt.current_lr)
            else:
                self.opt.current_lr = self.opt.learning_rate_REINFORCE

            # Set Network models as Training mode
            self.encoder.train()
            self.decoderPolicyGradient.train()

            # For each Iteration,
            for iter, (images, captions, lengths, imgids) in enumerate(self.trainloader):

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
                self.optimizer_critic.zero_grad()

                # Extract Image Features
                features = self.encoder(images)
                featuresVolatile = self.encoder(imagesVolatile)

                # Generate Sequence g_1:T ~ Policy(.|Image) without Ground Truth (Training)
                maxSequenceLength = max(lengths)
                outputs = self.decoderPolicyGradient(features, captions, states, maxSequenceLength, lengths, gt=False)

                # Calculate loss for Log Display
                loss = self.criterion(outputs.detach(), targets)

                # Monte Carlo Rollouts - Each Rollout has own LSTM (Stochastic Recurrent Sampling)
                actions_rollouts     = self.decoderPolicyGradient.getMonteCarloRollouts(featuresVolatile, statesVolatile, self.opt.MC_rollouts, maxSequenceLength, gt=False)

                # Convert MC Rollout Results (Actions, Embedding Vectors) to Sentences (without <start> word)
                predictions_rollouts = self.decoderPolicyGradient.getSentences(actions_rollouts, imgids, self.vocab)

                # DEBUG -----------------------------------------------------
                if 0:
                    torch.set_printoptions(edgeitems=100, linewidth=160)
                    for prediction in predictions_rollouts:
                        print(prediction)
                # -----------------------------------------------------------

                # Calculate Rewards - Evaluate COCO Metrics
                rewards_batch, rewards_rollouts, lang_stat_rollouts = self.decoderPolicyGradient.getRewardsRollouts(predictions_rollouts, self.opt.batch_size, self.opt.MC_rollouts, lengths, maxSequenceLength, self.coco_train, self.valid_train)

                # Calculate Rewards
                rewards, rewardsMax, rewardsMin, rewardsAvg = self.decoderPolicyGradient.getRewards(rewards_batch)

                # Training the Network using REINFORCE : Calculate loss and Optimize the Network
                #  - actions: list [torch.longTensor, torch.longTensor, ..., torch.longTensor]
                #  -  action: Variable[torch.LongTensor of size 1x1]
                #  - rewards: [torch.FloatTensor of size 31]
                #  -       r: float
                # ------------------------------------------------------------------------------
                # self.decoderPolicyGradient.actions: 19xbatch_size
                # rewards:
                for action, r in zip(self.decoderPolicyGradient.actions, rewards):
                    action.reinforce(r.unsqueeze(1))
                autograd.backward(self.decoderPolicyGradient.actions, [None for _ in self.decoderPolicyGradient.actions])
                clip_gradient(self.optimizer, self.opt.grad_clip)
                self.optimizer.step()

                # Training the Network of Baseline Estimator : Calculate loss and Optimize the Network
                #  - value  = self.critic_linear(hiddens.squeeze(1).detach())
                #  - self.values.append(value.mean())
                #  - nn.MSELOSS(self.decoderPolicyGradient.values[1:], x)
                # loss_critic = self.criterion_critic(torch.cat([_.mean() for _ in self.decoderPolicyGradient.values[1:]]).contiguous(), Variable(torch.Tensor(rewards_rollouts).cuda(), requires_grad=False))
                loss_critic = torch.mean(torch.pow(torch.stack(self.decoderPolicyGradient.values).squeeze() - Variable(rewards.cuda(), requires_grad=False), 2))
                loss_critic.backward()
                clip_gradient(self.optimizer_critic, self.opt.grad_clip)
                self.optimizer_critic.step()

                # Display Information :: REINFORCE
                if iter % self.opt.log_step == 0:
                    self.decoderPolicyGradient.displaySaveInformationREINFORCE(epoch, self.opt.max_epochs, iter, self.total_train_iter, loss, loss_critic, rewardsMin, rewardsAvg, rewardsMax, self.opt.current_lr, self.opt.expr_dir, self.opt.exp_id, predictions_rollouts, lang_stat_rollouts)

                # Delete Variables every iteration
                self.decoderPolicyGradient.deleteVariables()

            # Make evaluation on validation set, and save model
            val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt, self.coco_valid, self.valid_valid, self.opt.batch_size)
            # Display & Save Information :: MLE Show and Tell
            self.decoderPolicyGradient.displaySaveInformationCOCOMetric(self.opt.expr_dir, self.opt.exp_id, lang_stats, mode='REINFORCE')
            # Save the Pre-trained Model for Every Epoch
            torch.save(self.encoder.state_dict(),               os.path.join('model/', self.opt.exp_id + '_REINFORCE-encoder-epoch%d.pkl' % epoch))
            torch.save(self.decoderPolicyGradient.state_dict(), os.path.join('model/', self.opt.exp_id + '_REINFORCE-decoderPolicyGradient-epoch%d.pkl' % (epoch)))


if __name__ == '__main__':
    if 0:
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
        parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
        parser.add_argument('--hidden_size', type=int, default=512,help='dimension of lstm hidden states')
        parser.add_argument('--num_layers', type=int, default=1,help='number of layers in lstm')
        parser.add_argument('--num_epochs', type=int, default=5)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        args = parser.parse_args()
        print(json.dumps(vars(args), indent=2))
        main(args)