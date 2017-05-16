from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def language_eval(preds):
    import sys
    sys.path.append("coco-caption")
    annFile = '/home/myunggi/Repository/Data/COCO/annotations_captions/captions_val2014.json'

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    random.seed(time.time())
    tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))

    if not os.path.exists('cache'):
        os.makedirs('cache')

    with open('cache/'+tmp_name+'.json', 'w') as f:
        json.dump(preds_filt, f)

    json.dump(preds_filt, open('cache/'+tmp_name+'.json', 'w'))

    resFile = 'cache/'+tmp_name+'.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # delete the temp file
    os.system('rm ' +'cache/'+tmp_name+'.json')

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    return out

def evaluation(model, crit, loader, vocab, opt):

    model.eval()

    loss_sum = 0
    loss_evals = 0
    predictions = []

    check_duplicate = []
    caption_vis = True


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

        outputs = model(images, captions, lengths)

        loss = crit(outputs, targets)
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        sampled_ids = model.sample(images)

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

    if caption_vis:
        for ent in predictions[:10]:
            print("%s : %s" % (ent['image_id'], ent['caption']))

    lang_stats = language_eval(predictions)

    return loss_sum/loss_evals, predictions, lang_stats
