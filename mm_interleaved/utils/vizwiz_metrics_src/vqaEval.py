# coding=utf-8

__author__='Qing Li'

# This code is based on the code written by Qing Li for VizWiz Python API available at the following link: 
# (https://github.com/xxx)


import sys
import re
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

class VQAEval:
	def __init__(self, vqa, vqaRes, n=2):
		self.n 			  = n
		self.accuracy     = {}
		self.caption_metric = {}
		self.evalQA       = {}
		self.evalAnsType  = {}
		self.unanswerability = {}
		self.vqa 		  = vqa
		self.vqaRes       = vqaRes
		self.params		  = {'images': vqa.getImgs()}
		self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn’t’ve", "couldnt’ve": "couldn’t’ve", "didnt": "didn’t", "doesnt": "doesn’t", "dont": "don’t", "hadnt": "hadn’t", \
							 "hadnt’ve": "hadn’t’ve", "hadn'tve": "hadn’t’ve", "hasnt": "hasn’t", "havent": "haven’t", "hed": "he’d", "hed’ve": "he’d’ve", \
							 "he’dve": "he’d’ve", "hes": "he’s", "howd": "how’d", "howll": "how’ll", "hows": "how’s", "Id’ve": "I’d’ve", "I’dve": "I’d’ve", \
							 "Im": "I’m", "Ive": "I’ve", "isnt": "isn’t", "itd": "it’d", "itd’ve": "it’d’ve", "it’dve": "it’d’ve", "itll": "it’ll", "let’s": "let’s", \
							 "maam": "ma’am", "mightnt": "mightn’t", "mightnt’ve": "mightn’t’ve", "mightn’tve": "mightn’t’ve", "mightve": "might’ve", \
							 "mustnt": "mustn’t", "mustve": "must’ve", "neednt": "needn’t", "notve": "not’ve", "oclock": "o’clock", "oughtnt": "oughtn’t", \
							 "ow’s’at": "’ow’s’at", "’ows’at": "’ow’s’at", "’ow’sat": "’ow’s’at", "shant": "shan’t", "shed’ve": "she’d’ve", "she’dve": "she’d’ve", \
							 "she’s": "she’s", "shouldve": "should’ve", "shouldnt": "shouldn’t", "shouldnt’ve": "shouldn’t’ve", "shouldn’tve": "shouldn’t’ve", \
							 "somebody’d": "somebodyd", "somebodyd’ve": "somebody’d’ve", "somebody’dve": "somebody’d’ve", "somebodyll": "somebody’ll", \
							 "somebodys": "somebody’s", "someoned": "someone’d", "someoned’ve": "someone’d’ve", "someone’dve": "someone’d’ve", \
							 "someonell": "someone’ll", "someones": "someone’s", "somethingd": "something’d", "somethingd’ve": "something’d’ve", \
							 "something’dve": "something’d’ve", "somethingll": "something’ll", "thats": "that’s", "thered": "there’d", "thered’ve": "there’d’ve", \
							 "there’dve": "there’d’ve", "therere": "there’re", "theres": "there’s", "theyd": "they’d", "theyd’ve": "they’d’ve", \
							 "they’dve": "they’d’ve", "theyll": "they’ll", "theyre": "they’re", "theyve": "they’ve", "twas": "’twas", "wasnt": "wasn’t", \
							 "wed’ve": "we’d’ve", "we’dve": "we’d’ve", "weve": "we've", "werent": "weren’t", "whatll": "what’ll", "whatre": "what’re", \
							 "whats": "what’s", "whatve": "what’ve", "whens": "when’s", "whered": "where’d", "wheres": "where's", "whereve": "where’ve", \
							 "whod": "who’d", "whod’ve": "who’d’ve", "who’dve": "who’d’ve", "wholl": "who’ll", "whos": "who’s", "whove": "who've", "whyll": "why’ll", \
							 "whyre": "why’re", "whys": "why’s", "wont": "won’t", "wouldve": "would’ve", "wouldnt": "wouldn’t", "wouldnt’ve": "wouldn’t’ve", \
							 "wouldn’tve": "wouldn’t’ve", "yall": "y’all", "yall’ll": "y’all’ll", "y’allll": "y’all’ll", "yall’d’ve": "y’all’d’ve", \
							 "y’alld’ve": "y’all’d’ve", "y’all’dve": "y’all’d’ve", "youd": "you’d", "youd’ve": "you’d’ve", "you’dve": "you’d’ve", \
							 "youll": "you’ll", "youre": "you’re", "youve": "you’ve"}
		self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}
		self.articles     = ['a',
							 'an',
							 'the'
							]
 

		self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
		self.commaStrip   = re.compile("(\d)(\,)(\d)")
		self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']

	
	def evaluate(self, imgs=None):
		if imgs == None:
			imgs = [img for img in self.params['images']]
		gts = {}
		res = {}
		for img in imgs:
			gts[img] = self.vqa.imgToQA[img]
			res[img] = self.vqaRes.imgToQA[img]
		
		# =================================================
		# Compute accuracy
		# =================================================
		accQA       = []
		accAnsType  = {}
		print("computing accuracy")
		step = 0
		for img in imgs:
			resAns      = res[img]['answer']
			resAns      = resAns.replace('\n', ' ')
			resAns      = resAns.replace('\t', ' ')
			resAns      = resAns.strip()
			resAns      = self.processPunctuation(resAns)
			resAns      = self.processDigitArticle(resAns)
			gtAnswers = []
			gtAcc = [] 
			for i, ans in enumerate(gts[img]['answers']):
				otherGTAns = [item for j, item in enumerate(gts[img]['answers']) if i != j]
				matchingAns = [item for item in otherGTAns if item['answer']==resAns]
				acc = min(1, float(len(matchingAns))/3)
				gtAcc.append(acc)
			avgGTAcc = float(sum(gtAcc))/len(gtAcc)
			accQA.append(avgGTAcc)

			ansType     = gts[img]['answer_type']
			if ansType not in accAnsType:
				accAnsType[ansType] = []
			accAnsType[ansType].append(avgGTAcc)
			self.setEvalQA(img, avgGTAcc)
			self.setEvalAnsType(img, ansType, avgGTAcc)
			if step%1000 == 0:
				self.updateProgress(step/float(len(imgs)))
			step = step + 1

		self.setAccuracy(accQA, accAnsType)
		print("Done computing accuracy")

		# =================================================
		# Compute caption metric
		# =================================================
		print("computing caption metric")
		gts = {}
		res = {}
		for img in imgs:
			x = self.vqaRes.imgToQA[img]
			res[img] = [{'image_id':x['image'], 'caption':x['answer']}]

			gtAnswers = self.vqa.imgToQA[img]
			ans_list = [x['answer'] for x in gtAnswers['answers']]
			tmp = []
			for x in ans_list:
				try:
					tmp.append(str(x))
				except:
					pass
			ans_list = tmp
			ans_list = [{'image_id': img, 'caption': str(x)} for x in ans_list]
			gts[img] = ans_list

		evalObj = COCOEvalCap(list(gts.keys()), gts, res)
		evalObj.evaluate()
		for k, v in list(evalObj.eval.items()):
			self.caption_metric[k] = round(100*v, self.n)
		print("Done computing caption metric")
		
	def evaluate_unanswerability(self, imgs=None):
		if imgs == None:
			imgs = [img for img in self.params['images']]
		pred = []
		gt_labels = []
		for img in imgs:
			gt_labels.append(self.vqa.imgToQA[img]['answerable'])
			pred.append(self.vqaRes.imgToQA[img]['answerable'])
		gt_labels = np.array(gt_labels)
		pred = np.array(pred)
		
		gt_labels_n = 1 - gt_labels
		pred_n = 1.0 - pred
		average_precision = average_precision_score(gt_labels_n, pred_n)
		one_f1_score = f1_score(gt_labels_n, pred_n > 0.5)
		
		self.unanswerability['average_precision'] = round(100*average_precision, self.n)
		self.unanswerability['f1_score'] = round(100*one_f1_score, self.n)

	
	def processPunctuation(self, inText):
		outText = inText
		for p in self.punct:
			if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
				outText = outText.replace(p, '')
			else:
				outText = outText.replace(p, ' ')	
		outText = self.periodStrip.sub("",
									  outText,
									  re.UNICODE)
		return outText
	
	def processDigitArticle(self, inText):
		outText = []
		tempText = inText.lower().split()
		for word in tempText:
			word = self.manualMap.setdefault(word, word)
			if word not in self.articles:
				outText.append(word)
			else:
				pass
		for wordId, word in enumerate(outText):
			if word in self.contractions: 
				outText[wordId] = self.contractions[word]
		outText = ' '.join(outText)
		return outText

	def setAccuracy(self, accQA, accAnsType):
		self.accuracy['overall']         = round(100*float(sum(accQA))/len(accQA), self.n)
		self.accuracy['perAnswerType']   = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) for ansType in accAnsType}
			
	def setEvalQA(self, img, acc):
		self.evalQA[img] = round(100*acc, self.n)

	def setEvalAnsType(self, img, ansType, acc):
		if ansType not in self.evalAnsType:
			self.evalAnsType[ansType] = {}
		self.evalAnsType[ansType][img] = round(100*acc, self.n)

	def updateProgress(self, progress):
		barLength = 20
		status = ""
		if isinstance(progress, int):
			progress = float(progress)
		if not isinstance(progress, float):
			progress = 0
			status = "error: progress var must be float\r\n"
		if progress < 0:
			progress = 0
			status = "Halt...\r\n"
		if progress >= 1:
			progress = 1
			status = "Done...\r\n"
		block = int(round(barLength*progress))
		text = "\rFinshed Percent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), int(progress*100), status)
		# sys.stdout.write(text)
		# sys.stdout.flush()
		print(text)



from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

class COCOEvalCap:
    def __init__(self,images,gts,res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.params = {'image_id': images}
        self.gts = gts
        self.res = res

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = self.gts
        res = self.res

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            assert(set(gts.keys()) == set(res.keys()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in list(self.imgToEval.items())]
