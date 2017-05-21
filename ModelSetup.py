from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle

import torch
import torch.nn as nn

from models.ShowTellModel import ShowTellModel
from models.ShowAttendTellModel import ShowAttendTellModel

from models.models1 import EncoderCNN, DecoderRNN, DecoderPolicyGradient
from models.models2 import ShowAttendTellModel_XE
from models.models2 import ShowAttendTellModel_G, ShowAttendTellModel_D


def model_setup(opt, model_name):

    if model_name == 'show_tell':
        opt.load_pretrain = False
        opt.start_from = False
        model = ShowTellModel(opt)

    elif model_name == 'show_attend_tell':
        opt.load_pretrain = False
        opt.start_from = False
        model = ShowAttendTellModel(opt)

    elif model_name == 'ShowAttendTellModel_XE':
        opt.load_pretrain = False
        opt.start_from = False
        model = ShowAttendTellModel_XE(opt)

    else:
        raise Exception("Caption model not supported: {}".format(opt.model_name))

    if opt.num_gpu == 1:
        model = model.cuda()
    elif opt.num_gpu > 1:
        model = nn.DataParallel(model.cuda(), device_ids=range(opt.num_gpu))

    infos = {}
    if vars(opt).get('load_pretrain', False):
        pretrain_path = os.path.join(opt.root_dir, 'experiment', opt.user_id, opt.exp_id)
        assert os.path.isdir(pretrain_path), " %s must be a path" % pretrain_path
        assert os.path.isfile(os.path.join(pretrain_path, "infos-best.pkl")), "infos-best.pkl file does not exist in path %s" % pretrain_path
        model.load_state_dict(torch.load(os.path.join(pretrain_path, "model-best.pth")))
        print('load pretrained model from %s' % os.path.join(pretrain_path, "model-best.pth"))

    elif vars(opt).get('start_from', False):
        continue_path = os.path.join(opt.root_dir, 'experiment', opt.user_id, opt.exp_id)
        assert os.path.isdir(continue_path), " %s must be a path" % continue_path
        assert os.path.isfile(os.path.join(continue_path, "infos.pkl")), "infos.pkl file does not exist in path %s" % continue_path
        model.load_state_dict(torch.load(os.path.join(continue_path, "model.pth")))
        with open(os.path.join(continue_path, 'infos' + '.pkl')) as f:
            infos = pickle.load(f)
        print('continue training the model from %s' % os.path.join(continue_path, "model.pth"))
    else:
        print('training the %s model from scratch' % model_name)

    return model, infos


# For Trainer_GAN
def model_setup_2(opt, model_name):

    if model_name == 'ShowAttendTellModel_GAN':
        opt.load_pretrain = False
        opt.start_from = False
        model = [ShowAttendTellModel_G(opt), ShowAttendTellModel_D(opt)]

    else:
        raise Exception("Caption model not supported: {}".format(opt.model_name))

    if opt.num_gpu == 1:
        model = [m.cuda() for m in model]
    elif opt.num_gpu > 1:
        model = [nn.DataParallel(m.cuda(), device_ids=range(opt.num_gpu)) for m in model]

    infos = {}
    if vars(opt).get('load_pretrain', False):
        pretrain_path = os.path.join(opt.root_dir, 'experiment', opt.user_id, opt.exp_id)
        assert os.path.isdir(pretrain_path), " %s must be a path" % pretrain_path
        assert os.path.isfile(os.path.join(pretrain_path, "infos-best.pkl")), "infos-best.pkl file does not exist in path %s" % pretrain_path
        model[0].load_state_dict(torch.load(os.path.join(pretrain_path, "model-best.pth")))
        model[1].load_state_dict(torch.load(os.path.join(pretrain_path, "model-best.pth")))
        print('load pretrained model from %s' % os.path.join(pretrain_path, "model-best.pth"))

    elif vars(opt).get('start_from', False):
        continue_path = os.path.join(opt.root_dir, 'experiment', opt.user_id, opt.exp_id)
        assert os.path.isdir(continue_path), " %s must be a path" % continue_path
        assert os.path.isfile(os.path.join(continue_path, "infos.pkl")), "infos.pkl file does not exist in path %s" % continue_path
        model[0].load_state_dict(torch.load(os.path.join(continue_path, "model.pth")))
        model[1].load_state_dict(torch.load(os.path.join(continue_path, "model.pth")))
        with open(os.path.join(continue_path, 'infos' + '.pkl')) as f:
            infos = pickle.load(f)
        print('continue training the model from %s' % os.path.join(continue_path, "model.pth"))

    else:
        print('training the %s model from scratch' % model_name)

    return model, infos