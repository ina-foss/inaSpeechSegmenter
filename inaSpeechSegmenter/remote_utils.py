import os
from keras.utils import get_file

ISS_url = 'https://huggingface.co/0xrushi/inaSpeechSegmenter-torch/resolve/main/'
VFS_url = 'https://huggingface.co/0xrushi/inaSpeechSegmenter-torch/resolve/main/'
OLD_ISS_url = 'https://github.com/ina-foss/inaSpeechSegmenter/releases/download/models/'
OLD_VFS_url = 'https://github.com/ina-foss/inaSpeechSegmenter/releases/download/interspeech23/'

dmodels = {
    'keras_speech_music_cnn.pth': ISS_url,
    'keras_speech_music_noise_cnn.pth': ISS_url,
    'keras_male_female_cnn.pth': ISS_url,
    'interspeech2023_all.pth': VFS_url,
    'interspeech2023_cvfr.pth': VFS_url,
    'keras_speech_music_cnn.hdf5': OLD_ISS_url,
    'keras_speech_music_noise_cnn.hdf5': OLD_ISS_url,
    'keras_male_female_cnn.hdf5': OLD_ISS_url,
    'interspeech2023_all.hdf5': OLD_VFS_url,
    'interspeech2023_cvfr.hdf5': OLD_VFS_url,
    'final.onnx': 'https://github.com/ina-foss/inaSpeechSegmenter/releases/download/interspeech23/',
    'raw_81.pth': 'https://github.com/ina-foss/inaSpeechSegmenter/releases/download/interspeech23/',
}


def get_remote(model_fname):
    # if in a docker image, try to get the file in /root/.keras
    rootpath = '/root/.keras/inaSpeechSegmenter/' + model_fname
    if os.access(rootpath, os.R_OK):
        return rootpath

    # standard keras get file
    # check if the file is in /home/$USER/.keras/inaSpeechSegmenter/ and download it if required
    url = dmodels[model_fname]
    return get_file(model_fname, url + model_fname, cache_subdir='inaSpeechSegmenter')
