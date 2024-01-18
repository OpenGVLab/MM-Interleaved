__author__ = 'QingLi'
__version__ = '1.0'

# Interface for accessing the VQA dataset.

# This code is based on the code written by Qing Li for VizWiz Python API available at the following link:
# (https://github.com/xxx)

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime
import copy


class VQA:
    def __init__(self, annotation_file=None, annotation=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.imgToQA = {}
        if annotation is not None or annotation_file is not None:
            print('loading dataset into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r')) if annotation is None else annotation
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.imgToQA = {x['image']: x for x in dataset}

    def getImgs(self):
        return list(self.imgToQA.keys())

    def getAnns(self, imgs=[], ansTypes=[]):
        """
        Get annotations that satisfy given filter conditions. default skips that filter
        :param  imgs (str array): get annotations for given image names
                        ansTypes  (str array)   : get annotations for given answer types
        :return: annotations  (dict array)   : dict array of annotations
        """
        anns = self.dataset

        imgs = imgs if type(imgs) == list else [imgs]
        if len(imgs) != 0:
            anns = [self.imgToQA[img] for img in imgs]

        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]
        if len(ansTypes) != 0:
            anns = [ann for ann in anns if ann['answer_type'] in ansTypes]
        return anns

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            print("Question: %s" % ann['question'])
            print("Answer: ")
            print('\n'.join([x['answer'] for x in ann['answers']]))
