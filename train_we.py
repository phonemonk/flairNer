
import sys
import os
import logging

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, FlairEmbeddings
from flairNer import ner_trainer

logpath = "train_we.log"

logging.basicConfig(level=logging.DEBUG,
        filename=logpath,
        filemode='w')

PY = "PID:%d:PY::" % os.getpid()

logging.info(PY + "Starting Training  of Word Embedding...")

infilepath = sys.argv[1]
print("Infilepath: %s", infilepath)

print("Done")



