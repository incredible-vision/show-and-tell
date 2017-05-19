import os
import json
import pickle
import argparse
from datetime import datetime

def parse_opt():

    parser = argparse.ArgumentParser()

    # Input Data
    parser.add_argument('--root_dir',                           type=str,   default='/home/dehlix/Projects/Captioning/show-and-tell',     help="root directory")
    parser.add_argument('--data_json',                          type=str,   default='data/data.json',                                     help='input data list which includes captions and image information')
    parser.add_argument('--num_gpu',                            type=int,   default=1,                                                    help='number of gpus available, if set to 0, use cpu instead')
    parser.add_argument('--random_seed',                        type=int,   default=321,                                                  help='random seed number, to reproduce the result, fix the number')
    parser.add_argument('--crop_size',                          type=int,   default=224,                                                  help='image crop size, spatial dimension of input to the encoder')
    parser.add_argument('--batch_size',                         type=int,   default=128,                                                  help='batch size for training')
    parser.add_argument('--sentence_size',                      type=int,   default=20,                                                   help='maximum sentence size')
    parser.add_argument('--MC_rollouts',                        type=int,   default=3,                                                    help='Monte Carlo Rollouts')

    # Vocab
    parser.add_argument('--vocab_path',                         type=str,   default='data/vocab.pkl',                                     help='vocabulary wrapper')

    # Experiment Path
    parser.add_argument('--expr_dir',                           type=str,   default='experiment',                                         help='experiment directory')
    parser.add_argument('--exp_id',                             type=str,   default=datetime.today().strftime('%Y%m%d_%H%M%S'),           help='experiment id')
    parser.add_argument('--start_from',                         type=str,   default=None,                                                 help='continue from this configurations')

    # LSTM Model Parameters
    parser.add_argument('--embed_size',                         type=int,   default=512,                                                  help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size',                        type=int,   default=512,                                                  help='dimension of lstm hidden states')
    parser.add_argument('--num_layers',                         type=int,   default=1,                                                    help='number of layers in lstm')

    # Load Pre-trained Models & Saved Optimizers
    # parser.add_argument('--load_model_flag',                    type=bool,  default=False)
    # parser.add_argument('--load_model_path',                    type=str,   default=list(('model/Pre-trained-encoder-epoch31.pkl', 'model/Pre-trained-decoderPolicyGradient-epoch31.pkl')))
    parser.add_argument('--load_optim_flag',                    type=bool,  default=False)
    parser.add_argument('--load_optim_path',                    type=str,   default=None)

    # Learning Rate
    parser.add_argument('--learning_rate',                      type=float, default=5e-4)
    parser.add_argument('--learning_rate_REINFORCE',            type=float, default=5e-4)

    # Epoch
    parser.add_argument('--max_epochs',                         type=int,   default=30)
    parser.add_argument('--max_epochs_REINFORCE',               type=int,   default=30)
    parser.add_argument('--learning_rate_decay_start',          type=int,   default=1,                                                  help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every',          type=int,   default=1,                                                  help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate',           type=float, default=0.8,                                                help='every how many iterations thereafter to drop LR?(in epoch)')

    # Gradient Clip
    parser.add_argument('--grad_clip',                          type=float, default=0.1,                                                help='clip gradients at this value')

    # Scheduled Sampling
    parser.add_argument('--scheduled_sampling_start',           type=int,   default=-1,                                                 help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every',  type=int,   default=5,                                                  help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob',   type=float, default=0.05,                                               help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob',        type=float, default=0.25,                                               help='Maximum scheduled sampling prob.')

    # Display & Save Logs
    parser.add_argument('--log_step',                           type=int,   default=10,                                                 help='step size for printing log info')
    parser.add_argument('--save_checkpoint_every',              type=int,   default=1,                                                  help='how often to save a model checkpoint (in iterations)?')

    # ETC
    parser.add_argument('--load_best_score',                    action="store_true",   default=True)

    args = parser.parse_args()
    return args

# Save Configurations to the pre-defined location
def save_config(opt):
    # Check the existence of the Folder
    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)
    # Set Path
    save_path = os.path.join(opt.expr_dir, str(opt.exp_id) + "_config_expr" + ".pkl")
    # Save Configurations
    with open(save_path, 'wb') as f:
        print("[saving config file...], save to %s" % save_path)
        pickle.dump(opt, f)
