import os
import json
import pickle
import argparse

def parse_opt():

    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--root_dir', type=str, default='/home/myunggi/Research/CapGAN', help="root directory")
    parser.add_argument('--data_json', type=str, default='data/data.json', help='input data list which includes captions and image information')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='vocabulary wrapper')

    parser.add_argument('--num_gpu', type=int, default=1, help='number of gpus available, if set to 0, use cpu instead')
    parser.add_argument('--random_seed', type=int, default=123, help='random seed number, to reproduce the result, fix the number')
    parser.add_argument('--crop_size', type=int, default=224, help='image crop size, spatial dimension of input to the encoder')
    parser.add_argument('--image_size', type=int, default=224, help='image crop size, spatial dimension of input to the encoder')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')

    parser.add_argument('--expr_dir', type=str, default='experiment', help='experiment directory')
    parser.add_argument('--exp_id', type=str, default='cap_gan', help='experiment id')
    parser.add_argument('--user_id', type=str, default='myunggi', help='user id')
    parser.add_argument('--start_from', type=str, default=None, help='continue from this configurations')

    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--max_length', type=int, default=20, help='maximum length of sampling sequences')

    parser.add_argument('--load_best_score', action="store_true", default=True)
    parser.add_argument('--load_model_path', action="store_true", default=False)
    parser.add_argument('--load_optim_path', action="store_true", default=False)
    parser.add_argument('--load_pretrained', action='store_false', default=False)

    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--max_epochs', type=int, default=120)

    parser.add_argument('--learning_rate_decay_start', type=int, default=1, help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--grad_clip', type=float, default=0.5, help='clip gradients at this value')# 5.,

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, help='Maximum scheduled sampling prob.')

    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')

    parser.add_argument('--language_eval', type=int, default=1, help='1 for Cider score, 0 for log loss')

    parser.add_argument('--save_checkpoint_every', type=int, default=500, help='how often to save a model checkpoint (in iterations)?')

    args = parser.parse_args()

    return args

def save_config(opt):
    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)
    save_path = os.path.join(opt.expr_dir, "config_expr_"+ str(opt.exp_id)+".pkl")

    print("[saving config file...], save to %s" % save_path)

    with open(save_path, 'wb') as f:
        #pickle.dump(opt.__dict__, f, indent=4, sort_keys=True)
        pickle.dump(opt, f)