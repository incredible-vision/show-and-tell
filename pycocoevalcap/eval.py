__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
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
        if len(res) != 128:
            print('---------------------------------')
            print('gts')
            print(gts)
            print('')
            print('---------------------------------')
            print('res')
            print(res)
            print('')
            print(len(gts), len(res), len(imgIds))
            for idx in imgIds:
                if len(res[idx]) != 1:
                    tmp = res[idx][0]
                    res[idx] = []
                    res[idx].append(tmp)

        for scorer, method in scorers:
            # print('computing %s score...'%(scorer.method()))
            # Input
            #  - gts: {4032: ['an image of a row of buses parked in the parking lot', 'a large line of buses sitting in a row', 'row of five yellow school busses in parked position', 'several school buses are lined up and parked', 'five school buses are lined up in a parking lot'], 323498: ['a desk with several computers on it in the corner of a room', 'a large flat panel computer monitor on a desk', 'a desk is full of two laptops and a desktop computer', 'computer desk with different computers on top of it', 'a view of a desk with computers and books']}
            #  - res: {4032: ['some elephants are walking around in their business'], 323498: ['a person in black jacket over a corner table with computers']}
            # Output
            #  - score:   [0.4210526315346261, 0.1573778950558735, 1.18194899348602e-06, 3.3570917693817216e-09]
            #  - scores: [[0.24999999993750016, 0.545454545355372], [5.976143045124577e-09, 0.23354968320493186], [1.8123006211824564e-11, 1.8232183582936018e-06], [1.0445522727757714e-12, 5.246341021824447e-09]]
            score, scores = scorer.compute_score(gts, res)

            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

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
