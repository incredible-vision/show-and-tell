# --------------------------------------------------------------------------------
# Improved Image Captioning via Policy Gradient Optimization of SPIDEr (REINFORCE)
# --------------------------------------------------------------------------------

import os
import json
import pickle
import torch
from torchvision import transforms
from data_loader import get_loader
from utils import Vocabulary
from config import parse_opt, save_config
from train_SPIDEr import Trainer

def main(opt):

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

    # Initialize Data Loaders (Training & Validation)
    train_dataloader = get_loader(opt, mode='train', transform=train_transform)
    valid_dataloader = get_loader(opt, mode='val',   transform=valid_transform)
    print('Load the dataset into memory...')
    print('Total iterations in training phase : {} \ntotal iterations in validation phase : {}'.format(len(train_dataloader), len(valid_dataloader)))

    # Save Configurations to the pre-defined location
    save_config(opt)

    # Type #1. Pre-Train the Show and Tell Model using Maximum Likelihood Estimation on Training Dataset.
    if 0:
        # Declare the Trainer
        load_model_path = []
        trainer = Trainer(opt, train_dataloader, valid_dataloader, load_model_path, load_model_flag=False)
        # Start Training Network Models
        trainer.trainMLE()

    # Type #2. Evaluate COCO Metrics on Validation Dataset.
    if 0:
        # Declare the Trainer
        load_model_path = list(('model/Pre-trained-encoder-epoch20.pkl',
                                'model/Pre-trained-decoderPolicyGradient-epoch20.pkl'))
        trainer = Trainer(opt, train_dataloader, valid_dataloader, load_model_path, load_model_flag=True)
        # Start Validating Network Models
        trainer.validation()

    # 3. Train the SPIDEr Model using Policy Gradient (REINFORCE) on Training Dataset
    if 1:
        # Declare the Trainer
        load_model_path = list(('model/Pre-trained-encoder-epoch19.pkl',
                                'model/Pre-trained-decoderPolicyGradient-epoch19.pkl'))
        trainer = Trainer(opt, train_dataloader, valid_dataloader, load_model_path, load_model_flag=True)
        trainer.trainREINFORCE()

if __name__ == "__main__":
    print('Improved Image Captioning via Policy Gradient Optimization of SPIDEr (REINFORCE)')
    # Set Configurations
    args = parse_opt()
    # Do Main Function
    main(args)
