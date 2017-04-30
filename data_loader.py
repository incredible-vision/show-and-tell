import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import json
import pickle
import argparse
from PIL import Image
import numpy as np
from utils import Vocabulary

class CocoDataset(data.Dataset):

    def __init__(self, root, anns, vocab, mode='train',transform=None):

        self.root = root
        self.anns = json.load(open(anns))
        self.vocab = pickle.load(open(vocab, 'rb'))
        self.transform = transform

        self.data = [ann for ann in self.anns if ann['split'] == mode]


    def __getitem__(self, index):
        data  = self.data
        vocab = self.vocab
        # load image
        path = os.path.join(self.root, data[index]['file_path'])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # load caption
        cap = data[index]['final_caption']
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(word) for word in cap])
        caption.append(vocab('<end>'))

        target = torch.IntTensor(caption)

        return img, target, data[index]['imgid']

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    # sort the data in descending order
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, imgids = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths, imgids

def get_loader(opt, mode='train', shuffle=True, num_workers=1, transform=None):

    coco = CocoDataset(root=opt.root_dir,
                       anns=opt.data_json,
                       vocab=opt.vocab_path,
                       mode=mode,
                       transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='/home/myunggi/Research/show-and-tell', help="root directory of the project")
    parser.add_argument('--data_json', type=str, default='data/data.json', help='input data list which includes captions and image information')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='vocabulary wrapper')
    parser.add_argument('--crop_size', type=int, default=224, help='image crop size')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_loader = get_loader(args, transform=transform)
    total_iter = len(data_loader)
    for i, (img, target, length) in enumerate(data_loader):

        print('done')
