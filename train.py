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
from models import EncoderCNN, DecoderRNN
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from eval import evaluation


class Trainer(object):
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

        self.encoder = EncoderCNN(opt.embed_size)
        self.decoder = DecoderRNN(opt.embed_size, opt.hidden_size,
                                  len(self.vocab), opt.num_layers)

        if self.num_gpu == 1:
            self.encoder.cuda()
            self.decoder.cuda()

        elif self.num_gpu > 1:
            self.encoder = nn.DataParallel(self.encoder.cuda(), device_ids=range(self.num_gpu))
            self.decoder = nn.DataParallel(self.decoder.cuda(), device_ids=range(self.num_gpu))

        if self.load_model_path:
            self.load_model()

        if self.load_optimizer_path:
            self.load_optimizer()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(list(self.encoder.resnet.fc.parameters())+list(self.decoder.parameters()), lr=opt.learning_rate)

        print('done')

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

        for epoch in range(1, 1 + self.opt.max_epochs):

            if epoch > self.opt.learning_rate_decay_start and self.opt.learning_rate_decay_start >= 1:
                fraction = (epoch - self.opt.learning_rate_decay_start) // self.opt.learning_rate_decay_every
                decay_factor = self.opt.learning_rate_decay_rate ** fraction
                self.opt.current_lr = self.opt.learning_rate * decay_factor
                set_lr(self.optimizer, self.opt.current_lr)
            else:
                self.opt.current_lr = self.opt.learning_rate

            # Assign the scheduled sampling prob
            if epoch > self.opt.scheduled_sampling_start and self.opt.scheduled_sampling_start >= 0:
                fraction = (epoch - self.opt.scheduled_sampling_start) // self.opt.scheduled_sampling_increase_every
                self.opt.ss_prob = min(self.opt.scheduled_sampling_increase_prob * fraction, self.opt.scheduled_sampling_max_prob)
                self.decoder.ss_prob = self.opt.ss_prob

            for iter, (images, captions, lengths, imgids) in enumerate(self.trainloader):

                torch.cuda.synchronize()
                start = time.time()

                # Set mini-batch dataset
                images = Variable(images)
                captions = Variable(captions)

                if self.num_gpu > 0:
                    images = images.cuda()
                    captions = captions.cuda()

                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                # Forward, Backward and Optimize
                self.decoder.zero_grad()
                self.encoder.zero_grad()
                features = self.encoder(images)
                outputs = self.decoder(features, captions, lengths)
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

                # Write the training loss summary
                # if (iter % self.opt.losses_log_every == 0):
                #     self.loss_history[iter] = loss.data[0]
                #     self.lr_history[iter] = self.decoder.current_lr
                #     self.ss_prob_history[iter] = self.decoder.ss_prob

                # make evaluation on validation set, and save model
                if (iter % self.opt.save_checkpoint_every == 0):
                    val_loss, predictions, lang_stats = evaluation(self.encoder, self.decoder, self.criterion, self.validloader, self.vocab, self.opt)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='/home/gt/PycharmProjects/show-and-tell', help="root directory of the project")
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
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))
