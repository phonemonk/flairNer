FROM nvidia/cuda:10.1-devel-ubuntu18.04
#From python:3.6-slim
#From anibali/pytorch:no-cuda

RUN apt-get update && apt-get install -y unixodbc-dev gcc g++

RUN mkdir -p flairNER/
COPY * flairNER/
WORKDIR flairNER/

RUN pip install -r requirements.txt

#RUN python -c 'import flair; _ = flair.models.SequenceTagger.load("ner-fast")'

CMD /bin/sh
