import os
import json
import pickle
import argparse
import numpy as np
from random import seed
from collections import Counter
from scipy.misc import imread, imresize, imsave


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

def print_stats(words, sent_lengths, opt):
    # print some stats of the captioning annotations
    total_words = sum(words.values())
    bad_words = [w for w,n in words.items() if n <= opt.word_count_threshold]
    vocab = [w for w,n in words.items() if n > opt.word_count_threshold]
    bad_count = sum(words[w] for w in bad_words)

    print('\ntotal words:', total_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(words), len(bad_words) * 100.0 / len(words)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    max_len = max(sent_lengths.keys())
    sum_len = sum(sent_lengths.values())

    print('\nmax length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

def build_vocab(opt):

    # read json file
    if not os.path.exists(opt.caption_json):
        raise Exception("[!] {} not exists.".format(opt.caption_json))
    imgs = json.load(open(opt.caption_json))
    imgs = imgs['images']

    print('\nLoad coco annotations from %s' % opt.caption_json)

    # count up the number of words
    counter, sent_lengths = Counter(), {}
    for img in imgs:
        for sentence in img['sentences']:
            counter.update(sentence['tokens'])
            sent_lengths[len(sentence['tokens'])] = sent_lengths.get(len(sentence['tokens']), 0) + 1

    allwords = {word: cnt for word, cnt in counter.items()}
    if opt.print_stats : print_stats(allwords,sent_lengths, opt)

    words = [word for word, cnt in counter.items() if cnt >= opt.word_count_threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab, imgs

def resize_image(img_path, img_save_path, i, total_num):
    # load the image
    img = imread(img_path)
    try:
        Ir = imresize(img, (256,256))
    except:
        return False
    # handle grayscale input images
    if len(Ir.shape) == 2:
        Ir = Ir[:, :, np.newaxis]
        Ir = np.concatenate((Ir, Ir, Ir), axis=2)
    # save to local directory
    imsave(img_save_path, Ir)
    if i % 1000 == 0:
        print("[%d/%d] resizing and saving the image completed." % (i, total_num))
    return True

def main(opt):
    vocab, imgs = build_vocab(opt)

    print('Start resizing and saving images...\n')
    # create output json file
    out = []
    for i, img in enumerate(imgs):
        img_path = os.path.join(opt.images_root, img['filepath'], img['filename'])

        # save path for resized image, prepended to file_path in output json file
        img_save_path = os.path.join('data/', img['filepath'], img['filename'])
        if not os.path.exists(os.path.join('data/', img['filepath'])):
            os.makedirs(os.path.join('data/', img['filepath']))

        # store all useful information
        out_img = {}
        out_img['split'] = img['split']
        out_img['imgid'] = img['imgid']
        out_img['cocoid'] = img['cocoid']
        out_img['file_path'] = img_save_path

        # resize the image into (256,256) and save it in the data directory
        assert resize_image(img_path, img_save_path, i+1, len(imgs)), 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'])

        for i, sentence in enumerate(img['sentences']):
            out_img['final_caption'] = sentence['tokens']
            out_img['sentids'] = img['sentids'][i]
            out.append(out_img.copy())

    assert len(out) == 616767, "total captions in coco dataset dont's match, maybe your input json file is broken"
    print('\nResizing and saving images completed!\n')
    # sort the caption data file in ascending order
    out.sort(key=lambda x: len(x['final_caption']), reverse=False)
    # save the vocabulary and re-ordered data in directory

    pickle.dump(vocab, open(opt.vocab_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    json.dump(out, open(opt.output_json, 'w'))

    print("\nSaved the vocabulary wrapper to '%s'" % opt.vocab_path)
    print("Saved the annotation wrapper to '%s'" % opt.output_json)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input and output file configuration
    # parser.add_argument('--caption_json', required=True, help='input json file to create vocabulary')
    parser.add_argument('--caption_json', type=str, default='data/MSCOCO/annotations/dataset_coco.json', help='input json file to create vocabulary')
    parser.add_argument('--output_json', type=str, default='data/data.json', help ='save path for annotation json file with additional information')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for saving vocabulary wrapper')

    # options
    # parser.add_argument('--images_root', required=True, help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--images_root', type=str, default='data/MSCOCO', help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--print_stats', action="store_true", default=False, help='print out the stats of the mscoco captioning annotations')

    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    main(args)