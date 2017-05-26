import os
import json
import pickle
import torch
from torchvision import transforms
from DataLoader import get_loader
from Utils import Vocabulary
from Config import parse_opt, save_config
from Train import Trainer, Trainer_GAN
from Utils import setup_logging
import logging
import subprocess


def main(opt):

    if not os.path.exists(os.path.join('experiment', opt.user_id)):
        os.makedirs(os.path.join('experiment', opt.user_id))

    opt.expr_dir = os.path.join('experiment', opt.user_id, opt.exp_id)
    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    torch.manual_seed(opt.random_seed)
    if opt.num_gpu > 0:
        torch.cuda.manual_seed(opt.random_seed)

    train_transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    valid_transform = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        # transforms.RandomHorizontalFlip(), # do we need to flip when eval?
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataloader = get_loader(opt, mode='train', transform=train_transform)
    valid_dataloader = get_loader(opt, mode='val', transform=valid_transform)

    print('load the dataset into memory...')
    print('total iterations in training phase : {} \ntotal iterations in validation phase : {}'.format(len(train_dataloader), len(valid_dataloader)))

    #trainer = Trainer(opt, train_dataloader, valid_dataloader)
    #trainer = Trainer_PG(opt, train_dataloader, valid_dataloader)
    trainer = Trainer_GAN(opt, train_dataloader, valid_dataloader)

    trainer.train()
    print('done')


if __name__ == "__main__":
    args = parse_opt()

    setup_logging(os.path.join('log.txt'))
    logging.info("\nrun arguments: %s", json.dumps(vars(args), indent=4, sort_keys=True))

    main(args)
    print('done')