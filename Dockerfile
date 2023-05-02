# This docker image have been tested using the following configuration
# of nvidia drivers obtained using the command nvidia-smi
# NVIDIA-SMI 470.161.03    Driver Version: 470.161.03    CUDA Version: 11.4



FROM tensorflow/tensorflow:2.8.3-gpu-jupyter

MAINTAINER David Doukhan david.doukhan@gmail.com

RUN apt-get update \
    && apt-get install -y ffmpeg \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

# download models to be used by default
# this part is non mandatory, costs 15 Mo, and ease image usage
ARG u='https://github.com/ina-foss/inaSpeechSegmenter/releases/download/models/'
ADD ${u}keras_male_female_cnn.hdf5 \
    ${u}keras_speech_music_cnn.hdf5 \
    ${u}keras_speech_music_noise_cnn.hdf5 \
    /root/.keras/inaSpeechSegmenter/

# download models to use VoiceFemininityScoring
ARG u='https://github.com/ina-foss/inaSpeechSegmenter/releases/download/interspeech23/'
ADD ${u}interspeech2023_all.hdf5 \
    ${u}interspeech2023_cvfr.hdf5 \
    ${u}final.onnx \
    ${u}raw_81.pth \
    /root/.keras/inaSpeechSegmenter/

# make models available to non-root users
RUN chmod +x /root/
RUN chmod +r /root/.keras/inaSpeechSegmenter/*

WORKDIR /inaSpeechSegmenter
COPY . ./

RUN pip install --upgrade pip && pip install . && pip cache purge
