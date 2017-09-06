import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

from DataLoader import get_loader
from Utils import Vocabulary, visualize_loss
from ModelSetup import model_setup
from logger.logger import Logger

from Eval import evaluation
from Eval_SPIDEr import evaluationPolicyGradient, language_eval

#from models.models1 import EncoderCNN, DecoderRNN, DecoderPolicyGradient
#from models.models2 import ShowAttendTellModel_XE #, ShowAttendTellModel_PG
#from models.models2 import ShowAttendTellModel_G, ShowAttendTellModel_D


# Training using GAN + PG testing...
class Trainer_GAN(object):
    def __init__(self, opt, trainloader, validloader, mode=None):
        self.opt = opt
        self.seqlen = None

        self.trainloader, self.validloader = trainloader, validloader

        self.train_loss_win = None
        self.train_perp_win = None

        self.logger = Logger(self.opt)

        with open(opt.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
            opt.vocab_size = len(self.vocab)

        if mode == 'GAN_G_pretrain':
            self.model_G, self.model_D = model_setup(opt, model_name='ShowAttendTellModel_G_pretrain')

        elif mode == 'GAN_D_pretrain':
            self.model_G, self.model_D = model_setup(opt, model_name='ShowAttendTellModel_D_pretrain')

        elif mode == 'GAN_train':
            self.model_G, self.model_D = model_setup(opt, model_name='ShowAttendTellModel_GAN')
        else:
            raise Exception("Check the mode option please")

        self.max_length = 20

        """ This criterion combines LogSoftMax and NLLLoss in one single class """
        self.criterion_G = nn.CrossEntropyLoss(size_average=True)
        self.criterion_D = nn.CrossEntropyLoss(size_average=True)

        """ select RewardMode = <RollOut> or <NoRollOut>"""
        self.RewardMode = 'RollOut' # NoRollOut

        """ only update trainable parameters """
        G_parameters = filter(lambda p: p.requires_grad, self.model_G.parameters())
        D_parameters = filter(lambda p: p.requires_grad, self.model_D.parameters())
        self.G_optimizer = optim.Adam(G_parameters, lr=opt.learning_rate)
        self.D_optimizer = optim.Adam(D_parameters, lr=opt.learning_rate*1e-1)

        self.best_flag = False
        self.init_info()
        print('Initialized Trainer done.')

    ################ STEP 1. Generator -> MLE Pre-training (Show-tell Based model) ################

    def train_mle(self):

        for epoch in range(self.start_epoch + 1, 1 + self.opt.max_epochs):
            self.adjust_lr(epoch)
            self.model_G.train()

            for iter, (images, captions, lengths, imgids) in enumerate(self.trainloader):
                self.total_iter +=1
                torch.cuda.synchronize()

                images = Variable(images.cuda())
                captions = Variable(captions.cuda())

                lengths = [l-1 for l in lengths]
                targets = pack_padded_sequence(captions[:,1:], lengths, batch_first=True)[0]

                self.model_G.zero_grad()

                seqlen = self.seqlen if self.seqlen is not None else lengths[0]

                outputs = self.model_G(images, captions[:,:-1], seqlen, mode='trainMLE')
                outputs = self.convertOutputVariable(outputs, seqlen, lengths)

                loss = self.criterion_G(outputs, targets)
                loss.backward()

                self.clip_gradient(self.G_optimizer, self.opt.grad_clip)
                self.G_optimizer.step()
                torch.cuda.synchronize()

                if iter % self.opt.log_step == 0:
                    self.logger.MLE_Train_Logger(self.train_loss_history, len(self.trainloader), loss, epoch, iter)

                # Write the training loss summary
                self.loss_history[self.total_iter] = loss.data[0]
                self.lr_history[self.total_iter] = self.opt.current_lr

                # make evaluation on validation set, and save model
                if self.total_iter % self.opt.save_checkpoint_every == 0:
                    self.valid_mle()

    def valid_mle(self):
        print('start evaluate ...')
        val_loss, predictions, lang_stats = evaluation(self.model_G, self.criterion_G, self.validloader, self.vocab, self.opt)
        self.val_result_history[self.total_iter] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

        # Save model if is improving on validation result
        if self.opt.language_eval == 'CIDEr':
            current_score = lang_stats['CIDEr']
        elif self.opt.language_veal == 'LogLoss':
            current_score = - val_loss
        else:
            raise Exception('please check the language_eval type.')

        if self.best_val_score is None or current_score > self.best_val_score:
            self.best_val_score = current_score
            self.best_flag = True

        # Dump miscalleous informations
        self.dump_infos()
        self.save_mle()

    def save_mle(self):
        checkpoint_path = os.path.join(self.opt.expr_dir, 'model_mle_G.pth')
        torch.save(self.model_G.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(self.opt.expr_dir, 'optimizer_mle_G.pth')
        torch.save(self.G_optimizer.state_dict(), optimizer_path)

    ############################# STEP 2. Pretrain Discriminator model ############################

    def train_discriminator(self):

        for group in self.D_optimizer.param_groups:
            group['lr'] = 0.0001

        for epoch in range(self.start_epoch + 1, 1 + self.opt.max_epochs):

            self.model_D.train()
            self.model_G.train()

            for iter, (images, captions, lengths, cocoids) in enumerate(self.trainloader):

                self.total_iter += 1

                torch.cuda.synchronize()

                # Set mini-batch dataset
                images = Variable(images).cuda()
                captions = Variable(captions).cuda()

                self.model_D.zero_grad()

                ############################ Update D network ################################
                D_error = self.TrainDiscriminator(images, captions)
                ##############################################################################
                torch.cuda.synchronize()

                if iter % self.opt.log_step == 0:
                    self.logger.D_Train_Logger(self.train_loss_history, len(self.trainloader), D_error, epoch, iter)

                # make evaluation on validation set, and save model
                if self.total_iter % self.opt.save_checkpoint_every == 0:
                    self.valid_discriminator()

    def valid_discriminator(self):
        self.save_discriminator()

    def save_discriminator(self):
        checkpoint_path = os.path.join(self.opt.expr_dir, 'model_D_pretrained.pth')
        torch.save(self.model_D.state_dict(), checkpoint_path)

        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(self.opt.expr_dir, 'optimizer_D_pretrained.pth')
        torch.save(self.D_optimizer.state_dict(), optimizer_path)

    ############################# STEP 3. Training Adversarial model ##############################

    def train_adversarial(self):

        for epoch in range(self.start_epoch + 1, 1 + self.opt.max_epochs):

            self.adjust_lr(epoch)
            self.model_D.train()
            self.model_G.train()

            for iter, (images, captions, lengths, cocoids) in enumerate(self.trainloader):

                self.total_iter += 1
                torch.cuda.synchronize()

                # Set mini-batch dataset
                images = Variable(images).cuda()
                captions = Variable(captions).cuda()

                self.model_D.zero_grad()
                self.model_G.zero_grad()

                ############################ (1) Update D network ############################
                D_error = self.TrainDiscriminator(images, captions)

                ############################ (2) Update G network ############################
                if self.RewardMode == 'NoRollOut':
                    G_reward_avg = self.TrainGenerator_NoRollOutReward(images, captions)

                elif self.RewardMode == 'RollOut':
                    G_reward_avg = self.TrainGenerator_RollOutReward(images, captions)

                else:
                    raise NameError, 'plz check the self.RewardMode.. (enter NoRollOut or RollOut)'
                ##############################################################################

                torch.cuda.synchronize()

                if iter % self.opt.log_step == 0:
                    self.logger.Adversarial_Train_Logger(self.train_loss_history, len(self.trainloader), D_error, G_reward_avg, epoch, iter)

                # make evaluation on validation set, and save model
                if self.total_iter % self.opt.save_checkpoint_every == 0:
                    self.valid_Adversarial()

    def valid_Adversarial(self):
        val_loss, predictions, lang_stats = evaluation(self.model_G, self.criterion_G, self.validloader, self.vocab, self.opt)
        self.val_result_history[self.total_iter] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

        # Save model if is improving on validation result
        if self.opt.language_eval == 'CIDEr':
            current_score = lang_stats['CIDEr']
        else:
            current_score = - val_loss

        if self.best_val_score is None or current_score > self.best_val_score:
            self.best_val_score = current_score
            self.best_flag = True

    def save_Adversarial(self):
        checkpoint_path = os.path.join(self.opt.exp_dir, 'model_GAN_G.pth')
        torch.save(self.model_G.state_dict(), checkpoint_path)
        checkpoint_path = os.path.join(self.opt.exp_dir, 'model_GAN_D.pth')
        torch.save(self.model_D.state_dict(), checkpoint_path)

        print("model saved to {}".format(checkpoint_path))

        optimizer_path = os.path.join(self.opt.exp_dir, 'optimizer_GAN_G.pth')
        torch.save(self.G_optimizer.state_dict(), optimizer_path)
        optimizer_path = os.path.join(self.opt.exp_dir, 'optimizer_GAN_D.pth')
        torch.save(self.D_optimizer.state_dict(), optimizer_path)

    def TrainDiscriminator(self, images, captions):
        f_sentences, f_sentences_word_emb = self.model_G.sample_for_D(images, mode='greedy')  # ([128, 20])/ ([128,512]x20)
        f_sentences, f_sentences_word_emb = self.model_G.pad_after_EOS(f_sentences)

        r_sentences, r_sentences_word_emb = self.model_G.gt_sentences(captions)  # ([128, 20])/ ([128,512]x20)

        iter_batch_size = r_sentences.size()[0]  # instead of opt.batch_size

        f_label = Variable(torch.FloatTensor(iter_batch_size).cuda())
        r_label = Variable(torch.FloatTensor(iter_batch_size).cuda())
        f_label.data.fill_(0)
        r_label.data.fill_(1)

        f_D_output = self.model_D(f_sentences_word_emb, self.model_G.encoder(images).detach(), iter)
        f_error = self.criterion_D(f_D_output, f_label.long())
        f_error.backward()

        r_D_output = self.model_D(r_sentences_word_emb, self.model_G.encoder(images).detach(), iter)
        r_error = self.criterion_D(r_D_output, r_label.long())
        r_error.backward()

        D_error = f_error + r_error
        self.D_optimizer.step()

        return D_error

    def TrainGenerator_NoRollOutReward(self, images, captions):
        iter_batch_size = len(images)

        f_outputs_idx, f_outputs_actions, _ = self.model_G(images, captions, mode='trainGAN')
        f_outputs_idx, f_outputs_actions_embeddings = self.model_G.pad_after_EOS(f_outputs_idx)

        f_label = Variable(torch.FloatTensor(iter_batch_size).cuda())
        f_label.data.fill_(1)

        f_D_output = self.model_D(f_outputs_actions_embeddings, self.model_G.encoder(images).detach(), iter)
        f_outputs_idx = [f_output_idx.unsqueeze(1) for f_output_idx in f_outputs_idx.transpose(0, 1)]  # [20, 128]

        G_error_rewards = torch.cat(self.LossForEachBatch(f_D_output, f_label, mode='Acc'), 0).unsqueeze(1)
        G_error_rewards = G_error_rewards.data.cpu().numpy()
        G_reward_avg, G_rewards = np.average(G_error_rewards), torch.FloatTensor(G_error_rewards).cuda()
        G_rewards = np.repeat(G_rewards, 20)

        # (2) After End-token -> reward = 0 version..
        for action, idx, r in zip(f_outputs_actions, f_outputs_idx, G_rewards):
            tmp = ((idx.detach().data > 1).type('torch.cuda.FloatTensor') * r.expand(action.data.size()))
            action.reinforce(tmp)

        torch.autograd.backward(f_outputs_actions, [None for _ in f_outputs_actions])
        self.G_optimizer.step()

        return G_reward_avg

    def TrainGenerator_RollOutReward(self, images, captions):
        iter_batch_size = len(images)

        f_outputs_idx, f_outputs_actions, _ = self.model_G(images, captions, mode='trainGAN')
        f_outputs_idx, f_outputs_actions_embeddings = self.model_G.pad_after_EOS(f_outputs_idx)

        f_label = Variable(torch.FloatTensor(iter_batch_size).cuda())
        f_label.data.fill_(1)

        rollout_rewards = self.GetRollOutRewards(self.model_G, self.model_D, images, f_outputs_actions)  # [128, 20]
        rollout_rewards = [reward.unsqueeze(1).cuda() for reward in rollout_rewards.transpose(0, 1)]  # [20, 128]

        f_outputs_idx = [f_output_idx.unsqueeze(1) for f_output_idx in f_outputs_idx.transpose(0, 1)]  # [20, 128]

        for action, idx, r in zip(f_outputs_actions, f_outputs_idx, rollout_rewards):
            tmp = ((idx.detach().data > 1).type('torch.cuda.FloatTensor') * r.data)
            action.reinforce(tmp)

        torch.autograd.backward(f_outputs_actions, [None for _ in f_outputs_actions])

        G_reward_avg = [rewards.data.cpu().numpy() for rewards in rollout_rewards]
        G_reward_avg = np.average(np.array(G_reward_avg))
        self.G_optimizer.step()

        return G_reward_avg

    def LossForEachBatch(self, outputs, labels, mode):
        if mode == 'BCE':
            # output must have passed through F.SIGMOID
            loss = [-(torch.log(outputs[i]) * (labels[i]) + torch.log(1 - outputs[i]) * (1 - labels[i]))
                    for i in range(outputs.size(0))]
            return loss
        elif mode == 'NLL':
            # np.average(loss) == criterion_G(outputs) // check complete
            outputs = F.softmax(outputs)
            loss = [-(torch.log(outputs[i][1]) * labels[i] + torch.log(outputs[i][0]) * (1 - labels[i]))
                    for i in range(outputs.size(0))]
            return loss

        elif mode == 'Acc':
            loss = [outputs[i][1] for i in range(outputs.size(0))]
            return loss
        else:
            raise Exception('mode options must be BCE or NLL.')

    def GetRollOutRewards(self, model_G, model_D, images, f_outputs_actions):
        # [20 x [128, 1]]
        rewards_matrix = Variable(
            torch.FloatTensor(f_outputs_actions[0].size()[0], len(f_outputs_actions)))  # [128,20]

        max_len = 20
        for t in range(1, max_len + 1):
            input_sentences = f_outputs_actions[:t]
            rollout = model_G.sample_for_G_rollout(images, input_sentences, max_len)
            _, rollout_embeddings = model_G.pad_after_EOS(rollout)

            output = model_D(rollout_embeddings, model_G.encoder(images).detach(), iter)

            ''' Define Rewards'''
            reward = F.sigmoid(output)[:, 1]
            rewards_matrix[:, t - 1] = reward

        return rewards_matrix.detach()  # [128,20]

    def adjust_lr(self, epoch):
        if 1 <= self.opt.learning_rate_decay_start < epoch:
            fraction = (epoch - self.opt.learning_rate_decay_start) // self.opt.learning_rate_decay_every
            decay_factor = self.opt.learning_rate_decay_rate ** fraction
            self.opt.current_lr = self.opt.learning_rate * decay_factor
            self.set_lr(self.G_optimizer, self.opt.current_lr)
        else:
            self.opt.current_lr = self.opt.learning_rate

    def set_lr(self, optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-grad_clip, grad_clip)

    def convertOutputVariable(self, outputs, maxlen, lengths):
        outputs = torch.cat(outputs, 1).view(len(lengths), maxlen, -1)
        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        return outputs

    def init_info(self, pretrained=False):
        if pretrained == True:
            ''' load pretrained info dictionary '''
            self.infos = dict()
            pass
        else:
            self.infos = dict()

        self.start_epoch = self.infos.get('epoch', 0)
        self.iteration = self.infos.get('iteration', 0)
        self.total_iter = self.infos.get('total_iter', 0)
        self.train_loss_history = self.infos.get('train_loss_history', {})
        self.val_result_history = self.infos.get('val_result_history', {})
        self.loss_history = self.infos.get('loss_history', {})
        self.lr_history = self.infos.get('lr_history', {})
        self.best_val_score = self.infos.get('best_val_score', None)
        self.best_flag = self.infos.get('best_flag', False)

    def dump_infos(self, mode):
        self.infos['total_iter'] = self.total_iter
        self.infos['iter'] = self.iteration
        self.infos['epoch'] = self.epoch
        self.infos['best_val_score'] = self.best_val_score
        self.infos['opt'] = self.opt
        self.infos['val_result_history'] = self.val_result_history
        self.infos['loss_history'] = self.loss_history
        self.infos['lr_history'] = self.lr_history
        self.infos['train_loss_history'] = self.train_loss_history
        self.best_val_score = self.infos.get('best_val_score', None)

        if self.best_flag:
            if mode == 'MLE_pretrain':
                checkpoint_path = os.path.join(self.opt.expr_dir, 'model_mle_G-best.pth')
                torch.save(self.model_G.state_dict(), checkpoint_path)
                print("model saved to {}".format(self.opt.expr_dir))
                with open(os.path.join(self.opt.expr_dir, 'model_mle_infos' + '-best.pkl'), 'wb') as f:
                    pickle.dump(self.infos, f)

            elif mode == 'Adversarial_pretrain':
                checkpoint_path = os.path.join(self.opt.expr_dir, 'model_GAN-best-G.pth')
                torch.save(self.model_G.state_dict(), checkpoint_path)
                checkpoint_path = os.path.join(self.opt.expr_dir, 'model_GAN-best-D.pth')
                torch.save(self.model_D.state_dict(), checkpoint_path)
                print("model saved to {}".format(self.opt.expr_dir))
                with open(os.path.join(self.opt.expr_dir, 'infos_adversarial' + '-best.pkl'), 'wb') as f:
                    pickle.dump(self.infos, f)

            else:
                raise Exception('please checke the mode. it must be "MLE_pretrain" or "Adversarial_pretrain')


if __name__ == '__main__':

    print 'Done.'

