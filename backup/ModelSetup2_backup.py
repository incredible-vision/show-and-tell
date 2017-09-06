from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle

import torch
import torch.nn as nn

from models.model_GANnREINFORCE.GANnREINFORCE import ShowAttendTellModel_G, ShowAttendTellModel_D, ShowAttendTellModel_D_imgATT

# For Trainer_GAN
def model_setup(opt, model_name):

    if model_name == 'ShowAttendTellModel_G_pretrain':
        opt.load_pretrain = False
        opt.start_from = False
        model = [ShowAttendTellModel_G(opt), ShowAttendTellModel_D(opt)]

    elif model_name == 'ShowAttendTellModel_D_pretrain':
        opt.load_pretrain = True
        opt.start_from = False
        #model = [ShowAttendTellModel_G(opt), ShowAttendTellModel_D(opt)]
        model = [ShowAttendTellModel_G(opt), ShowAttendTellModel_D_imgATT(opt)]

    elif model_name == 'ShowAttendTellModel_GAN':
        opt.load_pretrain = True
        opt.start_from = False
        #model = [ShowAttendTellModel_G(opt), ShowAttendTellModel_D(opt)]
        model = [ShowAttendTellModel_G(opt), ShowAttendTellModel_D_imgATT(opt)]

    else:
        raise Exception("Caption model not supported: {}".format(opt.model_name))

    if opt.num_gpu == 1:
        model = [m.cuda() for m in model]
    elif opt.num_gpu > 1:
        model = [nn.DataParallel(m.cuda(), device_ids=range(opt.num_gpu)) for m in model]

    if opt.load_pretrain == True:
        pretrain_path = os.path.join(opt.root_dir, 'experiment', 'pretrained')
        assert os.path.isdir(pretrain_path), " %s must be a path" % pretrain_path
        assert os.path.isfile(os.path.join(pretrain_path, "model_mle_G_infos-best.pkl")), \
            "infos-best.pkl file does not exist in path %s" % pretrain_path

        if model_name == 'ShowAttendTellModel_D_pretrain': # only load Generator
            model[0].load_state_dict(torch.load(os.path.join(pretrain_path, "model_mle_G-best.pth")))
            print('load pretrained G_model from %s' % os.path.join(pretrain_path, "model_mle_G-best.pth"))
            continue_train_D = False
            if continue_train_D:
                model[1].load_state_dict(torch.load(os.path.join(pretrain_path, "model_D_pretrained.pth")))
                print('load pretrained D_model from %s' % os.path.join(pretrain_path, "model_D_pretrained.pth"))

        elif model_name == 'ShowAttendTellModel_GAN':      # both load Generator and Discriminator
            model[0].load_state_dict(torch.load(os.path.join(pretrain_path, "model_mle_G-best.pth")))
            print('load pretrained G_model from %s' % os.path.join(pretrain_path, "model_mle_G-best.pth"))
            model[1].load_state_dict(torch.load(os.path.join(pretrain_path, "model_D_pretrained.pth")))
            print('load pretrained D_model from %s' % os.path.join(pretrain_path, "model_D_pretrained.pth"))
    else:
        print('training the %s model from scratch' % model_name)

    return model