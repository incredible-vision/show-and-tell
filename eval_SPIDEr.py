from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import json
import simplejson as json
from json import encoder
import random
import string
import time
import os
import sys
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def language_eval(preds, coco, valids, NEval, batch_size):

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    random.seed(time.time())
    tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    # print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open('cache/'+tmp_name+'.json', 'w'))

    resFile = 'cache/'+tmp_name+'.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate(NEval, batch_size)

    # delete the temp file
    os.system('rm ' +'cache/'+tmp_name+'.json')

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    return out

def evaluation(encoder, decoder, crit, loader, vocab, opt, coco, valids):
    verbose = True
    val_images_use = -1
    lang_eval = 1

    encoder.eval()
    decoder.eval()


    loss_sum = 0
    loss_evals = 0
    predictions = []
    check_duplicate = []

    for iter, (images, captions, lengths, imgids) in enumerate(loader):
        torch.cuda.synchronize()
        start = time.time()

        # Set mini-batch dataset
        images = Variable(images, volatile=True)
        captions = Variable(captions, volatile=True)
        state = (Variable(torch.zeros(opt.num_layers, images.size(0), opt.hidden_size), volatile=True),
                 Variable(torch.zeros(opt.num_layers, images.size(0), opt.hidden_size), volatile=True))

        if opt.num_gpu > 0:
            images = images.cuda()
            captions = captions.cuda()
            state = [s.cuda() for s in state]

        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        features = encoder(images)
        outputs = decoder(features, captions, lengths)

        loss = crit(outputs, targets)
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        sampled_ids = decoder.sample(features, state)
        sampled_ids = sampled_ids.cpu().data.numpy()
        result_sentences = []
        for sentence_ids in sampled_ids:
            sampled_caption = []
            for word_id in sentence_ids:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                sampled_caption.append(word)
            sentence = ' '.join(sampled_caption)
            result_sentences.append(sentence)
        for i, sentence in enumerate(result_sentences):
            if imgids[i] in check_duplicate:
                continue
            else:
                check_duplicate.append(imgids[i])
            entry = {'image_id': imgids[i], 'caption': sentence}
            predictions.append(entry)

    lang_stats = language_eval(predictions)
    return loss_sum/loss_evals, predictions, lang_stats


def evaluationPolicyGradient(encoder, decoderPolicyGradient, crit, loader, vocab, opt, coco, valids, batch_size):

    # Set Network Model as Evaluation Mode
    encoder.eval()
    decoderPolicyGradient.eval()

    # Initialize Variables
    loss_sum = 0
    loss_evals = 0
    predictions = []
    check_duplicate = []

    for iter, (images, captions, lengths, imgids) in enumerate(loader):

        # Set mini-batch dataset
        images   =  Variable(images,   volatile=True)
        captions =  Variable(captions, volatile=True)
        states   = (Variable(torch.zeros(opt.num_layers, images.size(0), opt.hidden_size), volatile=True),
                    Variable(torch.zeros(opt.num_layers, images.size(0), opt.hidden_size), volatile=True))

        # Set Variables to Support CUDA Computations
        if opt.num_gpu > 0:
            images = images.cuda()
            captions = captions.cuda()
            states = [s.cuda() for s in states]

        # Pack-Padded Sequence for Ground Truth Sentence
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Extract Image Features
        features = encoder(images)

        # Get Generated Sequence (Scores)
        maxSequenceLength = max(lengths)
        outputs = decoderPolicyGradient(features, captions, states, maxSequenceLength, lengths, gt=True)

        # Calculate Loss using MLE : Calculate loss and Optimize the Network
        loss = crit(outputs, targets)
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        # Convert to Sentences
        sampled_ids = decoderPolicyGradient.sample(features, states)
        sampled_ids = sampled_ids.data.cpu().numpy()
        result_sentences = []
        for sentence_ids in sampled_ids:
            sampled_caption = []
            for word_id in sentence_ids:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                sampled_caption.append(word)
            sentence = ' '.join(sampled_caption)
            result_sentences.append(sentence)
        for i, sentence in enumerate(result_sentences):
            if imgids[i] in check_duplicate:
                continue
            else:
                check_duplicate.append(imgids[i])
            entry = {'image_id': imgids[i], 'caption': sentence}
            predictions.append(entry)

        # Delete Variables
        decoderPolicyGradient.deleteVariables()
        if 0:
            del decoderPolicyGradient.values[:]
            del decoderPolicyGradient.outputs[:]
            del decoderPolicyGradient.actions[:]
            del decoderPolicyGradient.inputs[:]
            del decoderPolicyGradient.states[:]

    # Evaluation Generated Sentences
    NEval = len(predictions)
    lang_stats = language_eval(predictions, coco, valids, NEval, batch_size)

    return loss_sum / loss_evals, predictions, lang_stats
