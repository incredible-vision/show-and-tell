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
from models.Discriminator import Discriminator
from models.Encoder import EncoderCNN, EncoderCNN_F

def model_setup(opt, model_name):

    if model_name == 'show_tell':
        opt.pretrain_path = os.path.join('experiment', opt.user_id, 'show_tell', 'model-decoder.pth')
        opt.start_from = None
        model = ShowTellModel(opt)
    elif model_name == 'show_attend_tell':
        opt.pretrain_path = False
        opt.start_from = False
        model = ShowAttendTellModel(opt)
    elif model_name == 'discriminator':
        # opt.pretrain_path = os.path.join('experiment', opt.user_id, 'show_tell', 'model-discriminator.pth')
        opt.pretrain_path = None
        opt.start_from = None
        model = Discriminator(opt)
    elif model_name == 'cnn_encoder':
        opt.pretrain_path = os.path.join('experiment', opt.user_id, 'show_tell', 'model-encoder.pth')
        opt.start_from = None
        opt.cnn_type = 'resnet'
        opt.img_embed_size = 512
        model = EncoderCNN(opt)
    elif model_name == 'cnn_encoder_feature':
        opt.pretrain_path = False
        opt.start_from = False
        model = EncoderCNN_F(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.model_name))

    if opt.num_gpu == 1:
        model = model.cuda()
    elif opt.num_gpu > 1:
        model = nn.DataParallel(model.cuda(), device_ids=range(opt.num_gpu))

    infos = {}
    if vars(opt).get('pretrain_path', None) is not None:

        assert os.path.isfile(opt.pretrain_path), "file does not exist in path %s" % opt.pretrain_path
        model.load_state_dict(torch.load(opt.pretrain_path))
        print('load pretrained model from %s' % opt.pretrain_path)

    elif vars(opt).get('start_from', None) is not None:

        assert os.path.isfile(opt.start_from), "infos.pkl file does not exist in path %s" % opt.start_from
        model.load_state_dict(torch.load(opt.start_from))
        with open(os.path.join(opt.start_from, 'infos' + '.pkl')) as f:
            infos = pickle.load(f)
        print('continue training the model from %s' % opt.continue_path)
    else:
        print('training the %s model from scratch' % model_name)

    return model, infos

