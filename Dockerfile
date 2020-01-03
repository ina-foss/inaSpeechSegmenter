# Distrib
FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get -y update
RUN apt-get install -y ffmpeg python3 python-virtualenv python3-pip

RUN pip3 install tensorflow-gpu inaSpeechSegmenter
