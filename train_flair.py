
import sys
import os
import logging

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


