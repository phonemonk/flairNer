FROM nvidia/cuda:10.1-devel-ubuntu18.04
#From python:3.6-slim
#From anibali/pytorch:no-cuda

RUN apt-get update && apt-get install -y unixodbc-dev gcc g++
RUN apt-get install -y python3 python3-dev python3-pip

RUN mkdir -p flairNER/
COPY requirements.txt flairNER/
WORKDIR flairNER/
#RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install -r requirements.txt


COPY * ./
RUN mkdir -p data/
RUN mkdir -p models/

#RUN python -c 'import flair; _ = flair.models.SequenceTagger.load("ner-fast")'

CMD /bin/sh
