import os
import time
import torch
import numpy as np
import datetime
from Utils import visualize_loss

class Logger(object):

    def __init__(self, opt):
        self.opt = opt
        self.train_loss_win = None


    def MLE_Train_Logger(self, train_loss_history, total_iter, loss, epoch, iter):

        train_loss_history[total_iter] = {'loss': loss.data[0], 'perplexity': np.exp(loss.data[0])}
        self.train_loss_win = visualize_loss(self.train_loss_win, train_loss_history, 'train_loss', 'loss')

        log_print = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'\
                    % (epoch, self.opt.max_epochs, iter, total_iter, loss.data[0], np.exp(loss.data[0]))

        with open(os.path.join(self.opt.exp_dir, self.opt.exp_type, self.opt.exp_id, 'log_mle_pretrain.txt'), 'a') as f:
            print(log_print)
            f.write(log_print)
            f.write('\n')

    def MLE_Valid_Logger(self):
        pass


    def D_Train_Logger(self, train_loss_history, total_iter, D_error, epoch, iter):

        train_loss_history[total_iter] = {'loss': D_error.data[0]}
        self.train_loss_win = visualize_loss(self.train_loss_win, train_loss_history, 'Discriminator Pretrain Loss', 'loss')
        log_print = '[%d/%d][%d/%d] Loss_D: %.4f' % (epoch, self.opt.max_epochs, iter, len(self.trainloader), D_error.data[0])

        with open(os.path.join(self.opt.exp_dir, self.opt.exp_type, self.opt.exp_id, 'log_D_pretrain.txt'), 'a') as f:
            print(log_print)
            f.write(log_print)
            f.write('\n')

    def D_Valid_Logger(self):
        pass

    def Adversarial_Train_Logger(self, train_loss_history, total_iter, D_error, G_reward_avg, epoch, iter):

        train_loss_history[total_iter] = {'loss': G_reward_avg}
        self.train_loss_win = visualize_loss(self.train_loss_win, train_loss_history, 'Adversarial train: G_rewards', 'loss')

        log_print = '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, self.opt.max_epochs, iter, total_iter, D_error.data[0], G_reward_avg)

        with open(os.path.join(self.opt.exp_dir, self.opt.exp_type, self.opt.exp_id, 'log_D_pretrain.txt'), 'a') as f:
            print(log_print)
            f.write(log_print)
            f.write('\n')

    def Adversarial_Valid_Logger(self):
        pass

