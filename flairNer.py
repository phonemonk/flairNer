#!/bin/python

#
# Flair NER: Named Entity recognition using Flair library by Zalando Research.
# Algo module created for Acharya by Vimal Menon
# 2019
#

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List

class ner_trainer(object):
    def __init__(self, dataFolder, trainFile): 
        self.dataFolder = dataFolder
        self.trainFile = trainFile
        self.columns = {0: 'text', 1: 'pos', 2: 'ner'}
        pass

    def LoadConll03(self, dataFolder, trainFile, testFile=None, devFile=None):
        self.corpus = ColumnCorpus(dataFolder, self.columns, trainFile, testFile, devFile) 
        print(self.corpus.train[0].to_tagged_string('ner'))
