# This docker image have been tested using the following configuration
# of nvidia drivers obtained using the command nvidia-smi
# NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.4



FROM tensorflow/tensorflow:2.8.3-gpu-jupyter

MAINTAINER David Doukhan david.doukhan@gmail.com

RUN apt-get update \
    && apt-get install -y ffmpeg \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /inaSpeechSegmenter
COPY . ./

RUN pip install --upgrade pip && pip install . && pip cache purge
