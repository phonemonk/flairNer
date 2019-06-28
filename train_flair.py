
import sys
import os
import logging

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, FlairEmbeddings
from flairNer import ner_trainer

logpath = "train_ner.log"

logging.basicConfig(level=logging.DEBUG,
        filename=logpath,
        filemode='w')

PY = "PID:%d:PY::" % os.getpid()

logging.info(PY + "Starting Training ...")

infile = sys.argv[1]
datafolder = "data"
nt = ner_trainer(datafolder, infile)

nt.LoadConll03(datafolder, infile)

nt.LoadEmbeddings([
    WordEmbeddings('glove'), 
    PooledFlairEmbeddings('news-forward', pooling='min'), 
    FlairEmbeddings('resources/taggers/language_model/best-lm.pt')
    #PooledFlairEmbeddings('news-backward', pooling='min')
    ])

nt.train("models/tagger1", learning_rate=0.9, batch_size=32, hidden_size=256, epochs=200)

print("Done")



