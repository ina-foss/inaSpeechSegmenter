# Build command bellow:
# docker build --build-arg username=$USER --build-arg uid=`id -u $USER` .
#
# This docker image have been tested using the following configuration
# of nvidia drivers obtained using the command nvidia-smi
# NVIDIA-SMI 418.56 Driver Version: 418.56 CUDA Version: 10.1
# NVIDIA-SMI 450.80.02 Driver Version: 450.80.02 CUDA Version: 11.0

# for more recent drivers, the following configuration may work
# NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2
# FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

RUN apt-get update
RUN apt-get install -y ffmpeg

RUN pip install inaspeechsegmenter

# $USER
ARG username
# id -u $USER
ARG uid

RUN useradd --uid $uid -U -m  -s /bin/bash $username
USER $username
ENV HOME /home/$username
