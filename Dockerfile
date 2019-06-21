FROM nvidia/cuda:10.1-devel-ubuntu18.04
#From python:3.6-slim
#From anibali/pytorch:no-cuda

RUN apt-get update && apt-get install -y unixodbc-dev gcc g++
RUN apt-get install -y python3 python3-dev python3-pip

RUN pip3 install -r requirements.txt

RUN mkdir -p flairNER/
COPY * flairNER/
WORKDIR flairNER/
RUN mkdir -p data/

#RUN python -c 'import flair; _ = flair.models.SequenceTagger.load("ner-fast")'

CMD /bin/sh
