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

<<<<<<< HEAD
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
=======
    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')
>>>>>>> cc0e98fb3cfd9f140046ad3d6679f59d919a4f65

    random.seed(time.time())
    tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
<<<<<<< HEAD
    with open('cache/'+tmp_name+'.json', 'w') as f:
        json.dump(preds_filt, f)
=======
    json.dump(preds_filt, open('cache/'+tmp_name+'.json', 'w'))
>>>>>>> cc0e98fb3cfd9f140046ad3d6679f59d919a4f65

    resFile = 'cache/'+tmp_name+'.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # delete the temp file
<<<<<<< HEAD
    os.system('rm ' +'cache/'+tmp_name+'.json')
=======
    os.system('rm ' +tmp_name+'.json')
>>>>>>> cc0e98fb3cfd9f140046ad3d6679f59d919a4f65

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    return out

<<<<<<< HEAD
def evaluation(model, crit, loader, vocab, opt):
=======
def evaluation(encoder, decoder, crit, loader, vocab, opt):
>>>>>>> cc0e98fb3cfd9f140046ad3d6679f59d919a4f65
    verbose = True
    val_images_use = -1
    lang_eval = 1

<<<<<<< HEAD
    model.eval()
=======
    encoder.eval()
    decoder.eval()
>>>>>>> cc0e98fb3cfd9f140046ad3d6679f59d919a4f65

    loss_sum = 0
    loss_evals = 0
    predictions = []
<<<<<<< HEAD
    check_duplicate = []
=======
>>>>>>> cc0e98fb3cfd9f140046ad3d6679f59d919a4f65

    for iter, (images, captions, lengths, imgids) in enumerate(loader):
        torch.cuda.synchronize()
        start = time.time()

        # Set mini-batch dataset
        images = Variable(images, volatile=True)
        captions = Variable(captions, volatile=True)
<<<<<<< HEAD
        state = (Variable(torch.zeros(images.size(0), opt.hidden_size), volatile=True),
                 Variable(torch.zeros(images.size(0), opt.hidden_size), volatile=True))
=======
        state = (Variable(torch.zeros(opt.num_layers, images.size(0), opt.hidden_size), volatile=True),
                 Variable(torch.zeros(opt.num_layers, images.size(0), opt.hidden_size), volatile=True))
>>>>>>> cc0e98fb3cfd9f140046ad3d6679f59d919a4f65

        if opt.num_gpu > 0:
            images = images.cuda()
            captions = captions.cuda()
<<<<<<< HEAD
            state = torch.stack([s.cuda() for s in state])

        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        outputs = model(images, captions, lengths)

        loss = crit(outputs, targets)
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        sampled_ids = model.sample(images, state)
=======
            state = [s.cuda() for s in state]

        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        sampled_ids = decoder.sample(features, state)
        loss = crit(outputs, targets)

        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

>>>>>>> cc0e98fb3cfd9f140046ad3d6679f59d919a4f65
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
<<<<<<< HEAD
            if imgids[i] in check_duplicate:
                continue
            else:
                check_duplicate.append(imgids[i])
            entry = {'image_id': imgids[i], 'caption': sentence}
            predictions.append(entry)

    lang_stats = language_eval(predictions)

    return loss_sum/loss_evals, predictions, lang_stats
=======
            entry = {'image_id': imgids[i], 'caption': sentence}
            predictions.append(entry)

        lang_stats = language_eval(predictions)


        print('done')
>>>>>>> cc0e98fb3cfd9f140046ad3d6679f59d919a4f65
