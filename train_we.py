
import sys
import os
import logging

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, FlairEmbeddings
from flairNer import ner_trainer, trainLanguage

logpath = "train_we.log"

logging.basicConfig(level=logging.DEBUG,
        filename=logpath,
        filemode='w')

PY = "PID:%d:PY::" % os.getpid()

logging.info(PY + "Starting Training  of Word Embedding...")

infilepath = "data/vocabulary/"
charPath = sys.argv[1]
print("Infilepath: %s" % infilepath)

trainPath = infilepath + "train/"

os.system("mkdir -p %s" % trainPath)
os.system("head -100 %s/sentences.txt > %s/test.txt" % (infilepath, infilepath))
os.system("tail -100 %s/sentences.txt > %s/valid.txt" % (infilepath, infilepath))
os.system("cp %s/sentences.txt %s/train.txt" % (infilepath, trainPath))

trainLang = trainLanguage(charPath)
trainLang.trainLanguage(infilepath)

print("Done")



