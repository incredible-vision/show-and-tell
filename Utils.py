
import os
import json
import logging
import numpy as np
from datetime import datetime
from visualizer import Visdom

vis = Visdom()

def setup_logging(log_file='log.txt'):

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def visualize_loss(win, loss_history, title='loss', item_name='loss'):
    opts_loss = dict(title=title)

    loss_hist_key = sorted(loss_history.keys())
    if len(loss_history) == 2:
        win = vis.line(np.array([l[item_name] for l in loss_history.values()]), np.array(loss_hist_key), opts=opts_loss)

    if len(loss_history) > 2:
        win = vis.updateTrace(X=np.array([loss_hist_key[-1]]),
                              Y=np.array([loss_history[loss_hist_key[-1]][item_name]]),
                              win=win)
    return win

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)