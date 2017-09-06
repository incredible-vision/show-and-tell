import os
import json

from DataLoader import init_dataloader
from Utils import Vocabulary

from Config import parse_opt
from Trainer import Trainer_GAN

def main(opt):

    dataloader = init_dataloader(opt)

    #trainer = Trainer_GAN(opt, dataloader['train'], dataloader['val'], mode='GAN_G_pretrain')
    #trainer.train_mle()

    #trainer = Trainer_GAN(opt, dataloader['train'], dataloader['val'], mode='GAN_D_pretrain')
    #trainer.train_discriminator()

    trainer = Trainer_GAN(opt, dataloader['train'], dataloader['val'], mode='GAN_train')
    trainer.train_adversarial()

    print('done')


if __name__ == "__main__":

    opt = parse_opt()
    main(opt)

    print('done')