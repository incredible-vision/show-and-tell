import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import pickle
import json
from DataLoader import get_loader
from Utils import Vocabulary, visualize_loss
from ModelSetup import model_setup
from models.ShowTellModel import ShowTellModel
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from Eval import evaluation

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
        self.model, self.infos = model_setup(opt, model_name='show_tell')

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

class GAN_Trainer(object):
    """ Gan"""

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='/home/myunggi/Research/show-and-tell', help="root directory of the project")
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

    # main(args)

