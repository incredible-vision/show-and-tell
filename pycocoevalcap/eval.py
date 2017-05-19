__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
import numpy as np

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self, NEval, batch_size):
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        if 0:
            gts = {}
            res = {}
            for imgId in imgIds:
                gts[imgId] = self.coco.imgToAnns[imgId]
            for imgId in imgIds:
                res[imgId] = self.cocoRes.imgToAnns[imgId]
            gts  = tokenizer.tokenize(gts)
            res = tokenizer.tokenize(res)


        # =================================================
        # Set up scorers
        # =================================================
        # print('setting up scorers...')
        scorers = [
            (Bleu(4),  ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(),  "ROUGE_L"),
            (Cider(),  "CIDEr"),
            # (Spice(), "SPICE")
        ]
        # =================================================
        # Compute scores
        # =================================================
        # Modify gts and res to Evaluate COCO Metric with Vectorization
        if len(imgIds) < 512:  # Training Only!
            print(len(gts), len(res), len(imgIds))
            gts, res = self.modifygtsres(gts, res, imgIds, NEval, batch_size)
        print(len(gts), len(res), len(imgIds))

        # For Batch Error, You need to change batch size on the if statement.
        if 0:
            if tmp != batch_size:
                res = self.correctBatchError(gts, res, imgIds)
            print(len(gts), len(res), len(imgIds))

        # Evaluate COCO Metrics
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    if len(imgIds) < 512:  # Training Only!
                        self.setEval(scs, m)
                    else:  # Validation Only
                        self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.3f"%(m, sc))
            else:
                if len(imgIds) < 512:  # Training Only
                    self.setEval(scores, method)
                else:  # Validation Only
                    self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.3f"%(method, score))
        # self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(sorted(imgIds), scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [self.imgToEval[imgId] for imgId in sorted(self.imgToEval.keys())]


    def correctBatchError(self, gts, res, imgIds):
        print('Expected batch size is not the same as the number of sampled sets.')
        print(len(gts), len(res), len(imgIds))
        for idx in res.keys():
            if len(res[idx]) != 1:
                print(idx, res[idx])
                tmp = res[idx][0]
                res[idx] = []
                res[idx].append(tmp)
        return res

    def modifygtsres(self, gts, res, imgIds, NEval, batch_size):
        for imgId in imgIds:  # opt.batch_size
            resIdx = imgId * 1000
            tmp_res = res[imgId]
            if int(len(tmp_res)) != int(NEval/batch_size):
                if len(tmp_res) > int(NEval/batch_size):
                    print('Batch Size: 255 Error')
                    print('len(res[imgId]):%d' % (len(res[imgId])))
                    print('NEval:%d, batch_size:%d, int(NEval/batch_size): %d)' % (NEval, batch_size, int(NEval / batch_size)))
                    print(tmp_res)
                else:
                    print('len(res[imgId]):%d' % (len(res[imgId])))
                    print('NEval:%d, batch_size:%d, int(NEval/batch_size): %d)' % (NEval, batch_size, int(NEval / batch_size)))
                    print(tmp_res)
                    print('----------------------------')
            for idx in range(min(len(tmp_res), int(NEval/batch_size))):
                res[resIdx] = []
                res[resIdx].append(tmp_res[idx])
                resIdx += 1
            del res[imgId]
        for imgId in imgIds:  # opt.batch_size
            gtsIdx = imgId * 1000
            for idx in range(0, (len(res)/len(imgIds))):
                gts[gtsIdx] = {}
                gts[gtsIdx] = gts[imgId]
                gtsIdx += 1
            del gts[imgId]
        return gts, res
