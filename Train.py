import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import pickle
import json

from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from DataLoader import get_loader
from Utils import Vocabulary, visualize_loss
from ModelSetup import model_setup

from Eval import evaluation
from Eval_SPIDEr import evaluationPolicyGradient, language_eval

#from models.models1 import EncoderCNN, DecoderRNN, DecoderPolicyGradient
#from models.models2 import ShowAttendTellModel_XE #, ShowAttendTellModel_PG
#from models.models2 import ShowAttendTellModel_G, ShowAttendTellModel_D

class Trainer(object):
    def __init__(self, opt, trainloader, validloader):
        self.opt = opt
        self.num_gpu = opt.num_gpu
        self.seqlen = None

        self.total_train_iter = len(trainloader)
        self.total_valid_iter = len(validloader)

        self.trainloader = trainloader
        self.validloader = validloader

        self.train_loss_win = None
        self.train_perp_win = None

        with open(opt.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        opt.vocab_size = len(self.vocab)

        """ setup model and infos for training """
        self.model, self.infos = model_setup(opt, model_name='ShowAttendTellModel_XE')

        """ This criterion combines LogSoftMax and NLLLoss in one single class """
        self.criterion = nn.CrossEntropyLoss()

        """ only update trainable parameters """
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(parameters, lr=opt.learning_rate)

        if opt.start_from :
            self.optimizer.load_state_dict(torch.load(os.path.join(self.opt.expr_dir, 'optimizer.pth')))

        print('done')

    def train(self):

        total_iteration = self.infos.get('total_iter', 0)
        loaded_iteration = self.infos.get('iter', 0)
        loaded_epoch = self.infos.get('epoch', 0)
        val_result_history = self.infos.get('val_result_history', {})
        loss_history = self.infos.get('loss_history', {})
        lr_history = self.infos.get('lr_history', {})
        train_loss_history = self.infos.get('train_loss_history', {})

        # loading a best validation score
        if self.opt.load_best_score == True:
            best_val_score = self.infos.get('best_val_score', None)

        def convertOutputVariable(outputs, maxlen, lengths):
            outputs = torch.cat(outputs, 1).view(len(lengths), maxlen, -1)
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
            return outputs

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

        def set_lr(optimizer, lr):
            for group in optimizer.param_groups:
                group['lr'] = lr

        for epoch in range(loaded_epoch + 1, 1 + self.opt.max_epochs):

            if epoch > self.opt.learning_rate_decay_start and self.opt.learning_rate_decay_start >= 1:
                fraction = (epoch - self.opt.learning_rate_decay_start) // self.opt.learning_rate_decay_every
                decay_factor = self.opt.learning_rate_decay_rate ** fraction
                self.opt.current_lr = self.opt.learning_rate * decay_factor
                set_lr(self.optimizer, self.opt.current_lr)
            else:
                self.opt.current_lr = self.opt.learning_rate

            self.model.train()

            for iter, (images, captions, lengths, imgids) in enumerate(self.trainloader):

                iter += 1
                total_iteration += 1

                torch.cuda.synchronize()
                start = time.time()

                # Set mini-batch dataset
                images = Variable(images)
                captions = Variable(captions)

                if self.num_gpu > 0:
                    images = images.cuda()
                    captions = captions.cuda()

                lengths = [l-1 for l in lengths]
                targets = pack_padded_sequence(captions[:,1:], lengths, batch_first=True)[0]
                # Forward, Backward and Optimize
                self.model.zero_grad()

                # Sequence Length, we can manually designate maximum sequence length
                # or get maximum sequence length in ground truth captions
                seqlen = self.seqlen if self.seqlen is not None else lengths[0]

                outputs = self.model(images, captions[:,:-1], seqlen)
                outputs = convertOutputVariable(outputs, seqlen, lengths)

                loss = self.criterion(outputs, targets)
                loss.backward()
                clip_gradient(self.optimizer, self.opt.grad_clip)
                self.optimizer.step()

                torch.cuda.synchronize()
                end = time.time()

                if iter % self.opt.log_step == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                          % (epoch, self.opt.max_epochs, iter, self.total_train_iter,
                             loss.data[0], np.exp(loss.data[0])))
                    train_loss_history[total_iteration] = {'loss': loss.data[0], 'perplexity': np.exp(loss.data[0])}
                    self.train_loss_win = visualize_loss(self.train_loss_win, train_loss_history, 'train_loss', 'loss')

                # make evaluation on validation set, and save model
                if (total_iteration % self.opt.save_checkpoint_every == 0):
                    print('start evaluate ...')
                    val_loss, predictions, lang_stats = evaluation(self.model, self.criterion,
                                                                   self.validloader, self.vocab, self.opt)
                    val_result_history[total_iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                                           'predictions': predictions}

                    # Write the training loss summary
                    # loss_history[total_iteration] = loss.data[0].cpu().numpy()[0]
                    loss_history[total_iteration] = loss.data[0]
                    lr_history[total_iteration] = self.opt.current_lr

                    # Save model if is improving on validation result
                    if self.opt.language_eval == 1:
                        current_score = lang_stats['CIDEr']
                    else:
                        current_score = - val_loss

                    best_flag = False
                    if best_val_score is None or current_score > best_val_score:
                        best_val_score = current_score
                        best_flag = True

                    checkpoint_path = os.path.join(self.opt.expr_dir, 'model.pth')
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    optimizer_path = os.path.join(self.opt.expr_dir, 'optimizer.pth')
                    torch.save(self.optimizer.state_dict(), optimizer_path)

                    # Dump miscalleous informations
                    self.infos['total_iter'] = total_iteration
                    self.infos['iter'] = iter
                    self.infos['epoch'] = epoch
                    self.infos['best_val_score'] = best_val_score
                    self.infos['opt'] = self.opt
                    self.infos['val_result_history'] = val_result_history
                    self.infos['loss_history'] = loss_history
                    self.infos['lr_history'] = lr_history
                    self.infos['train_loss_history'] = train_loss_history
                    with open(os.path.join(self.opt.expr_dir, 'infos' + '.pkl'), 'wb') as f:
                        pickle.dump(self.infos, f)

                    if best_flag:
                        checkpoint_path = os.path.join(self.opt.expr_dir, 'model-best.pth')
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print("model saved to {}".format(self.opt.expr_dir))
                        with open(os.path.join(self.opt.expr_dir, 'infos' + '-best.pkl'), 'wb') as f:
                            pickle.dump(self.infos, f)


# Improved Image Captioning via Policy Gradient Optimization of SPIDEr (REINFORCE by SE)
class Trainer_PG(object):
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
        self.annFile_train = '/home/gt/D_Data/COCO/annotations_captions/captions_train2014.json'
        self.coco_train = COCO(self.annFile_train)
        self.valids_train = self.coco_train.getImgIds()
        self.annFile_valid = '/home/gt/D_Data/COCO/annotations_captions/captions_val2014.json'
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
            self.encoder.load_state_dict(torch.load('models/Pre-trained-encoder-epoch20.pkl'))
            self.decoderPolicyGradient.load_state_dict(torch.load('models/Pre-trained-decoderPolicyGradient-epoch20.pkl'))

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
                        print('[Loss: MLE] Epoch [%2d/%2d], Step [%4d/%4d], Loss: %2.4f, Perplexity: %6.4f, lr: %1.1e'
                              % (epoch, self.opt.max_epochs, iter, self.total_train_iter, loss.data[0], np.exp(loss.data[0]), self.opt.current_lr))

                    # Delete Variables
                    del self.decoderPolicyGradient.outputs[:]
                    del self.decoderPolicyGradient.actions[:]

                # Make evaluation on validation set, and save model
                val_loss, predictions, lang_stats = evaluationPolicyGradient(self.encoder, self.decoderPolicyGradient, self.criterion, self.validloader, self.vocab, self.opt, self.coco_valid, self.valids_valid)
                print(lang_stats)

                # Save the Pre-trained Model for Every Epoch
                torch.save(self.encoder.state_dict(),               os.path.join('models/', 'Pre-trained-encoder-epoch%d.pkl'               % (epoch)))
                torch.save(self.decoderPolicyGradient.state_dict(), os.path.join('models/', 'Pre-trained-decoderPolicyGradient-epoch%d.pkl' % (epoch)))

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

                    if len(imgids) < self.opt.batch_size:
                        print('Selected Batch size(%d) is not same as batch size' % len(imgids))
                        print(imgids)
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
                    maxSequenceLength = 15
                    _ = self.decoderPolicyGradient(features, captions, states, maxSequenceLength, lengths, gt=False)

                    # Monte Carlo Rollouts #1 - Every Rollout shares only one LSTM
                    # Monte Carlo Rollouts #2 - Each Rollout has own LSTM
                    actions_rollouts = self.decoderPolicyGradient.getMonteCarloRollouts(featuresVolatile, statesVolatile, self.opt.MC_rollouts, maxSequenceLength, gt=False)
                    predictions      = self.decoderPolicyGradient.getSentences(actions_rollouts, imgids, self.vocab)

                    if 0:
                        torch.set_printoptions(edgeitems=100, linewidth=160)
                        print(predictions)

                    # Calculate Rewards - Evaluate COCO Metrics
                    rewards_rollouts, lang_stat_rollouts = self.decoderPolicyGradient.getRewardsRollouts(predictions, self.opt.MC_rollouts, lengths, maxSequenceLength, self.coco_train, self.valids_train)

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
                        print('[REINFORCE] Epoch [%2d/%2d], Step [%4d/%4d], Rewards[min/avg/max]: [%.4f/%.4f/%.4f], Perplexity: [%6.4f/%6.4f/%6.4f], lr: %1.1e'
                              % (epoch, self.opt.max_epochs, iter, self.total_train_iter, rewardsMin, rewardsAvg, rewardsMax, np.exp(rewardsMin), np.exp(rewardsAvg), np.exp(rewardsMax), self.opt.current_lr))
                        log_print = '[REINFORCE] Epoch [%2d/%2d], Step [%4d/%4d], Rewards[min/avg/max]: [%.4f/%.4f/%.4f], Perplexity: [%6.4f/%6.4f/%6.4f], lr: %1.1e' % \
                                    (epoch, self.opt.max_epochs, iter, self.total_train_iter, rewardsMin, rewardsAvg, rewardsMax, np.exp(rewardsMin), np.exp(rewardsAvg), np.exp(rewardsMax), self.opt.current_lr)
                        with open('log.txt', 'a') as f:
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
                torch.save(self.encoder.state_dict(),               os.path.join('models/', 'REINFORCE-encoder-epoch%d.pkl'               % (epoch)))
                torch.save(self.decoderPolicyGradient.state_dict(), os.path.join('models/', 'REINFORCE-decoderPolicyGradient-epoch%d.pkl' % (epoch)))


class Trainer_GAN(object):
    def __init__(self, opt, trainloader, validloader):
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

        self.model_G = ShowAttendTellModel_G(opt.hidden_size, opt.embed_size, len(self.vocab), opt.embed_size, opt)
        self.model_D = ShowAttendTellModel_D(opt.hidden_size, opt.embed_size, len(self.vocab), opt.embed_size, opt)

        if self.num_gpu == 1:
            self.model_G.cuda()
            self.model_D.cuda()

        elif self.num_gpu > 1:
            self.model_G = nn.DataParallel(self.model_G.cuda(), device_ids=range(self.num_gpu))
            self.model_D = nn.DataParallel(self.model_D.cuda(), device_ids=range(self.num_gpu))

        if self.opt.load_pretrained:
            if self.load_model_path:
                self.load_model()
                print('pretrained model loaded')

            if self.load_optimizer_path:
                self.load_optimizer()

        """ This criterion combines LogSoftMax and NLLLoss in one single class """
        self.criterion = nn.CrossEntropyLoss()

        """ only update trainable parameters """
        G_parameters = filter(lambda p: p.requires_grad, self.model_G.parameters())
        D_parameters = filter(lambda p: p.requires_grad, self.model_D.parameters())
        self.G_optimizer = optim.Adam(G_parameters, lr=opt.learning_rate)
        self.D_optimizer = optim.Adam(D_parameters, lr=opt.learning_rate)

        print('done')

    def load_model(self):
        return self.model.load_state_dict(torch.load(os.path.join(self.opt.expr_dir, 'model-best.pth')))

    def load_optimizer(self):
        return self.optimizer.load_state_dict(torch.load(os.path.join(self.opt.expr_dir, 'optimizer.pth')))

    def train(self):

        infos = {}

        if self.opt.start_from is not None and not self.opt.load_pretrained:
            # open old infos and check if models are compatible
            with open(os.path.join(self.opt.expr_dir, 'infos' + '.pkl')) as f:
                infos = pickle.load(f)

        total_iteration = infos.get('total_iter', 0)
        loaded_iteration = infos.get('iter', 0)
        loaded_epoch = infos.get('epoch', 1)
        val_result_history = infos.get('val_result_history', {})
        loss_history = infos.get('loss_history', {})
        lr_history = infos.get('lr_history', {})

        # loading a best validation score
        if self.opt.load_best_score == True:
            best_val_score = infos.get('best_val_score', None)

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

        def set_lr(optimizer, lr):
            for group in optimizer.param_groups:
                group['lr'] = lr


        for epoch in range(1, 1 + self.opt.max_epochs):

            if epoch > self.opt.learning_rate_decay_start and self.opt.learning_rate_decay_start >= 1:
                fraction = (epoch - self.opt.learning_rate_decay_start) // self.opt.learning_rate_decay_every
                decay_factor = self.opt.learning_rate_decay_rate ** fraction
                self.opt.current_lr = self.opt.learning_rate * decay_factor
                set_lr(self.optimizer, self.opt.current_lr)
            else:
                self.opt.current_lr = self.opt.learning_rate

            if epoch < loaded_epoch: continue

            for iter, (images, captions, lengths, cocoids, imgids) in enumerate(self.trainloader):

                iter += 1
                total_iteration += 1
                if iter <= loaded_iteration: continue

                torch.cuda.synchronize()

                # Set mini-batch dataset
                images = Variable(images).cuda()
                captions = Variable(captions).cuda()

                lengths = [l-1 for l in lengths]
                targets = pack_padded_sequence(captions[:,1:], lengths, batch_first=True)[0]

                ############################
                # (1) Update D network:
                ###########################
                self.model_D.zero_grad()
                f_sentences_, f_sentences = self.model_G.sample(images)
                r_sentences = self.model_G.gt_sentences(captions, lengths)

                f_label, r_label = Variable(torch.FloatTensor(self.opt.batch_size).cuda()), Variable(torch.FloatTensor(self.opt.batch_size).cuda())
                f_label.data.resize_(self.opt.batch_size).fill(0)
                r_label.data.resize_(self.opt.batch_size).fill(1)

                f_output = self.model_D(f_sentences.detach())
                f_error = self.criterion(f_output, f_label)
                f_error.backward()

                r_output = self.model_D(r_sentences)
                r_error = self.criterion(r_output, r_label)
                r_error.backward()

                D_error = f_error + r_error
                self.D_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                self.model_G.zero_grad()
                '3: f_sentences, actions = self.model_G.sample(images)'
                f_label = Variable(torch.FloatTensor(self.opt.batch_size).cuda())
                f_label = f_label.data.resize_(self.opt.batch_size).fill(1)

                output = self.model_D(f_sentences.detach())
                G_error_rewards = self.criterion(output, f_label)

                '4)for action, r in zip(actions, G_error_rewards):'
                '   action.reinforce(r)'

                autograd.backward(actions, [None for _ in actions])
                self.optimizer_G.step()

                # Forward, Backward
                torch.cuda.synchronize()

                if iter % self.opt.log_step == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f'
                          .format(epoch, self.opt.max_epochs, iter, len(self.trainloader), D_error.data[0], G_error.data[0]))

                # make evaluation on validation set, and save model
                if (total_iteration % self.opt.save_checkpoint_every == 0):
                    val_loss, predictions, lang_stats = evaluation(self.model, self.criterion,
                                                                   self.validloader, self.vocab, self.opt)
                    val_result_history[total_iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                                           'predictions': predictions}

                    # Write the training loss summary
                    # loss_history[total_iteration] = loss.data[0].cpu().numpy()[0]
                    loss_history[total_iteration] = loss.data[0]
                    lr_history[total_iteration] = self.opt.current_lr

                    # Save model if is improving on validation result
                    if self.opt.language_eval == 1:
                        current_score = lang_stats['CIDEr']
                    else:
                        current_score = - val_loss

                    best_flag = False
                    if best_val_score is None or current_score > best_val_score:
                        best_val_score = current_score
                        best_flag = True

                    checkpoint_path = os.path.join(self.opt.expr_dir, 'model.pth')
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    optimizer_path = os.path.join(self.opt.expr_dir, 'optimizer.pth')
                    torch.save(self.optimizer.state_dict(), optimizer_path)

                    # Dump miscalleous informations
                    infos['total_iter'] = total_iteration
                    infos['iter'] = iter
                    infos['epoch'] = epoch
                    infos['best_val_score'] = best_val_score
                    infos['opt'] = self.opt
                    infos['val_result_history'] = val_result_history
                    infos['loss_history'] = loss_history
                    infos['lr_history'] = lr_history
                    with open(os.path.join(self.opt.expr_dir, 'infos' + '.pkl'), 'wb') as f:
                        pickle.dump(infos, f)

                    if best_flag:
                        checkpoint_path = os.path.join(self.opt.expr_dir, 'model-best.pth')
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print("model saved to {}".format(self.opt.expr_dir))
                        with open(os.path.join(self.opt.expr_dir, 'infos' + '-best.pkl'), 'wb') as f:
                            pickle.dump(infos, f)


if __name__ == '__main__':

    print 'Done.'