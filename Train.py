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
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from Eval import evaluation
from visualizer import Visdom

vis = Visdom()

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

        self.real_label = 1
        self.fake_label = 0

        with open(opt.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        opt.vocab_size = len(self.vocab)

        """ setup model and infos for training """
        self.encoder, _ = model_setup(opt, model_name='cnn_encoder')
        self.generator, self.infos_gen = model_setup(opt, model_name='show_tell')
        self.discriminator, self.infos_disc = model_setup(opt, model_name='discriminator')

        """ This criterion combines LogSoftMax and NLLLoss in one single class """
        self.criterion_MLE = nn.CrossEntropyLoss()
        self.criterion_G = nn.CrossEntropyLoss()
        self.criterion_D = nn.CrossEntropyLoss()

        """ only update trainable parameters """
        parameters = list(self.generator.parameters()) + list(self.encoder.parameters())
        parameters = filter(lambda p: p.requires_grad, parameters)
        # parameters = filter(lambda p: p.requires_grad, self.generator.parameters())
        self.optimizerM = optim.Adam(parameters, lr=opt.learning_rate)

        parameters = filter(lambda p: p.requires_grad, self.generator.parameters())
        self.optimizerG = optim.Adam(parameters, lr=opt.learning_rate)

        parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        self.optimizerD = optim.Adam(parameters, lr=5e-6)

        if opt.start_from :
            continue_path = os.path.join(opt.root_dir, 'experiment', opt.user_id, opt.exp_id,'optimizer.pth')
            self.optimizerM.load_state_dict(torch.load(continue_path))

        print('done')

    def train_mle(self):

        loaded_epoch = self.infos_gen.get('epoch', 0)
        loaded_iteration = self.infos_gen.get('iter', 0)
        lr_history = self.infos_gen.get('lr_history', {})
        loss_history = self.infos_gen.get('loss_history', {})
        total_iteration = self.infos_gen.get('total_iter', 0)

        val_result_history = self.infos_gen.get('val_result_history', {})
        train_loss_history = self.infos_gen.get('train_loss_history', {})

        # loading a best validation score
        if self.opt.load_best_score == True:
            best_val_score = self.infos_gen.get('best_val_score', None)


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
                set_lr(self.optimizerM, self.opt.current_lr)
            else:
                self.opt.current_lr = self.opt.learning_rate

            self.encoder.train()
            self.generator.train()

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
                self.encoder.zero_grad()
                self.generator.zero_grad()
                self.optimizerM.zero_grad()
                # Sequence Length, we can manually designate maximum sequence length
                # or get maximum sequence length in ground truth captions
                seqlen = self.seqlen if self.seqlen is not None else lengths[0]

                features = self.encoder(images)
                outputs = self.generator(features, captions[:,:-1], seqlen)
                outputs = convertOutputVariable(outputs, seqlen, lengths)

                loss = self.criterion_MLE(outputs, targets)
                loss.backward()
                clip_gradient(self.optimizerM, self.opt.grad_clip)
                self.optimizerM.step()

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
                    print('start evaluating ...')
                    val_loss, predictions, lang_stats = evaluation((self.encoder, self.generator), self.criterion_MLE,
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

                    checkpoint_path = os.path.join(self.opt.expr_dir, 'model-encoder.pth')
                    torch.save(self.encoder.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    checkpoint_path = os.path.join(self.opt.expr_dir, 'model-decoder.pth')
                    torch.save(self.generator.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    optimizer_path = os.path.join(self.opt.expr_dir, 'optimizer.pth')
                    torch.save(self.optimizerM.state_dict(), optimizer_path)
                    print("optimizer saved to {}".format(optimizer_path))

                    # Dump miscalleous informations
                    self.infos_gen['total_iter'] = total_iteration
                    self.infos_gen['iter'] = iter
                    self.infos_gen['epoch'] = epoch
                    self.infos_gen['best_val_score'] = best_val_score
                    self.infos_gen['opt'] = self.opt
                    self.infos_gen['val_result_history'] = val_result_history
                    self.infos_gen['loss_history'] = loss_history
                    self.infos_gen['lr_history'] = lr_history
                    self.infos_gen['train_loss_history'] = train_loss_history
                    with open(os.path.join(self.opt.expr_dir, 'infos' + '.pkl'), 'wb') as f:
                        pickle.dump(self.infos_gen, f)

                    if best_flag:
                        checkpoint_path = os.path.join(self.opt.expr_dir, 'model-encoder-best.pth')
                        torch.save(self.encoder.state_dict(), checkpoint_path)
                        print("model saved to {}".format(checkpoint_path))
                        checkpoint_path = os.path.join(self.opt.expr_dir, 'model-decoder-best.pth')
                        torch.save(self.generator.state_dict(), checkpoint_path)
                        print("model saved to {}".format(checkpoint_path))
                        with open(os.path.join(self.opt.expr_dir, 'infos' + '-best.pkl'), 'wb') as f:
                            pickle.dump(self.infos_gen, f)

    def train_discriminator(self):
        """"""
        total_iteration = self.infos_disc.get('total_iter', 0)
        loaded_iteration = self.infos_disc.get('iter', 0)
        loaded_epoch = self.infos_disc.get('epoch', 0)
        val_result_history = self.infos_disc.get('val_result_history', {})
        loss_history = self.infos_disc.get('loss_history', {})
        lr_history = self.infos_disc.get('lr_history', {})
        train_loss_history = self.infos_disc.get('train_loss_history', {})

        if self.opt.load_best_score == True:
            best_val_score = self.infos_disc.get('best_val_score', None)

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
                set_lr(self.optimizerD, self.opt.current_lr)
            else:
                self.opt.current_lr = self.opt.learning_rate

            self.discriminator.train()
            # self.generator.eval()

            for iter, (images, captions, lengths, imgids) in enumerate(self.trainloader):

                iter += 1
                total_iteration += 1

                #############################################################
                # Pretrain Discriminator Network
                #############################################################

                self.discriminator.zero_grad()

                # train with real
                # image_inputs.data.resize_(images.size()).copy_(images)
                # image_inputs.data.resize_(image_inputs.size()).copy_(images)
                images = Variable(images, volatile=False)
                sentence_real = Variable(captions, volatile=False)
                # labels.data.resize_(labels.size()).fill_(self.real_label)
                labels = Variable(torch.zeros(images.size(0)).fill_(self.real_label)).long()
                # images = Variable(images)
                # labels = Variable(labels)

                if self.num_gpu > 0:
                    images = images.cuda()
                    sentence_real = sentence_real.cuda()
                    labels = labels.cuda()

                # torch.cuda.synchronize()
                # start = time.time()

                logit = self.discriminator(sentence_real)
                loss_real = self.criterion_D(logit, labels.long())
                loss_real.backward()


                # train with fake
                features = self.encoder(images)
                sentence_fake = self.generator.sample(features.detach(), maxlen=20)
                sentence_fake = sentence_fake.detach()

                labels.data.fill_(self.fake_label)
                if self.num_gpu > 0:
                    sentence_fake = sentence_fake.cuda()
                    # labels = labels.cuda()

                logit = self.discriminator(sentence_fake)
                loss_fake = self.criterion_D(logit, labels.long())
                loss_fake.backward()
                total_loss = loss_real + loss_fake
                # total_loss = loss_real
                # torch.cuda.synchronize()
                # end = time.time()
                #
                self.optimizerD.step()


                if iter % self.opt.log_step == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                          % (epoch, self.opt.max_epochs, iter, self.total_train_iter,
                             total_loss.data[0]))
                    train_loss_history[total_iteration] = {'loss': total_loss.data[0]}
                    self.train_loss_win = visualize_loss(self.train_loss_win, train_loss_history, 'train_discriminator_loss', 'loss')

            checkpoint_path = os.path.join(self.opt.expr_dir, 'model-discriminator.pth')
            torch.save(self.discriminator.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))

    def train_adversarial(self):

        """"""
        total_iteration = self.infos_disc.get('total_iter', 0)
        loaded_iteration = self.infos_disc.get('iter', 0)
        loaded_epoch = self.infos_disc.get('epoch', 0)
        val_result_history = self.infos_disc.get('val_result_history', {})
        loss_history = self.infos_disc.get('loss_history', {})
        lr_history = self.infos_disc.get('lr_history', {})
        train_loss_history = self.infos_disc.get('train_loss_history', {})

        if self.opt.load_best_score == True:
            best_val_score = self.infos_disc.get('best_val_score', None)

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
                set_lr(self.optimizerD, self.opt.current_lr)
            else:
                self.opt.current_lr = self.opt.learning_rate

            self.discriminator.train()

            for iter, (images, captions, lengths, imgids) in enumerate(self.trainloader):

                iter += 1
                total_iteration += 1

                #############################################################
                # Update Discriminator Network
                #############################################################
                self.encoder.zero_grad()
                self.discriminator.zero_grad()
                self.generator.zero_grad()

                # train with real
                # image_inputs.data.resize_(images.size()).copy_(images)
                images = Variable(images, volatile=False)
                sentence_real = Variable(captions, volatile=False)
                labels = Variable(torch.zeros(images.size(0)).fill_(self.real_label)).long()

                if self.num_gpu > 0:
                    images = images.cuda()
                    sentence_real = sentence_real.cuda()
                    labels = labels.cuda()

                logit = self.discriminator(sentence_real)
                loss_real = self.criterion_D(logit, labels.long())
                loss_real.backward()

                # train with fake
                features = self.encoder(images)
                sentence_fake = self.generator.sample(features, maxlen=20)
                sentence_fake = sentence_fake.detach()

                labels.data.fill_(self.fake_label)
                if self.num_gpu > 0:
                    sentence_fake = sentence_fake.cuda()
                    labels = labels.cuda()

                logit = self.discriminator(sentence_fake)
                loss_fake = self.criterion_D(logit, labels.long())
                loss_fake.backward()
                total_loss = loss_real + loss_fake

                self.optimizerD.step()

                #############################################################
                # Update Generator Network
                #############################################################
                features = self.encoder(images)
                sentence_fake, actions = self.generator.sample_reinforce(features, maxlen=20)
                # sentence_fake = sentence_fake.detach()

                labels.data.fill_(self.real_label)
                # if self.num_gpu > 0:
                #     sentence_fake = sentence_fake.cuda()
                #     labels = labels.cuda()

                logit = self.discriminator(sentence_fake)
                loss_adversarial = self.criterion_G(logit, labels)
                rewards = np.repeat(loss_adversarial, sentence_fake.size(1))
                for action, r in zip(actions, rewards):
                    action.reinforce(float(r.data.cpu().numpy()[0]))
                autograd.backward(actions, [None for _ in actions])

                self.optimizerG.step()

                if iter % self.opt.log_step == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Discriminator Loss: %.4f, Generator Loss: %.4f'
                          % (epoch, self.opt.max_epochs, iter, self.total_train_iter,
                             total_loss.data[0], loss_adversarial.data[0]))
                    # train_loss_history[total_iteration] = {'loss': total_loss.data[0]}
                    # self.train_loss_win = visualize_loss(self.train_loss_win, train_loss_history, 'train_discriminator_loss', 'loss')
                if (total_iteration % self.opt.save_checkpoint_every == 0):
                    print('start evaluating ...')
                    val_loss, predictions, lang_stats = evaluation((self.encoder, self.generator), self.criterion_MLE,
                                                                   self.validloader, self.vocab, self.opt)
                    val_result_history[total_iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                                           'predictions': predictions}

                    # Write the training loss summary
                    # loss_history[total_iteration] = loss.data[0].cpu().numpy()[0]
                    loss_history[total_iteration] = total_loss.data[0]
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

                    checkpoint_path = os.path.join(self.opt.expr_dir, 'model-encoder.pth')
                    torch.save(self.encoder.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    checkpoint_path = os.path.join(self.opt.expr_dir, 'model-decoder.pth')
                    torch.save(self.generator.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    optimizer_path = os.path.join(self.opt.expr_dir, 'optimizer.pth')
                    torch.save(self.optimizerM.state_dict(), optimizer_path)
                    print("optimizer saved to {}".format(optimizer_path))

                    # Dump miscalleous informations
                    self.infos_gen['total_iter'] = total_iteration
                    self.infos_gen['iter'] = iter
                    self.infos_gen['epoch'] = epoch
                    self.infos_gen['best_val_score'] = best_val_score
                    self.infos_gen['opt'] = self.opt
                    self.infos_gen['val_result_history'] = val_result_history
                    self.infos_gen['loss_history'] = loss_history
                    self.infos_gen['lr_history'] = lr_history
                    self.infos_gen['train_loss_history'] = train_loss_history
                    with open(os.path.join(self.opt.expr_dir, 'infos' + '.pkl'), 'wb') as f:
                        pickle.dump(self.infos_gen, f)

                    if best_flag:
                        checkpoint_path = os.path.join(self.opt.expr_dir, 'model-encoder-best.pth')
                        torch.save(self.encoder.state_dict(), checkpoint_path)
                        print("model saved to {}".format(checkpoint_path))
                        checkpoint_path = os.path.join(self.opt.expr_dir, 'model-decoder-best.pth')
                        torch.save(self.generator.state_dict(), checkpoint_path)
                        print("model saved to {}".format(checkpoint_path))
                        with open(os.path.join(self.opt.expr_dir, 'infos' + '-best.pkl'), 'wb') as f:
                            pickle.dump(self.infos_gen, f)





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

