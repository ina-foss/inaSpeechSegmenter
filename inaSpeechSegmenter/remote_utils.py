import os
import torch

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
    # 1. Check Docker pre-downloaded path
    rootpath = '/root/.keras/inaSpeechSegmenter/' + model_fname
    if os.access(rootpath, os.R_OK):
        return rootpath

    # 2. Check Legacy Keras cache (~/.keras/inaSpeechSegmenter/)
    home = os.path.expanduser('~')
    keras_cache = os.path.join(home, '.keras', 'inaSpeechSegmenter', model_fname)
    if os.path.exists(keras_cache):
        return keras_cache

    # 3. Check/Download to Torch Hub cache (~/.cache/torch/hub/inaSpeechSegmenter/)
    hub_dir = torch.hub.get_dir()
    target_dir = os.path.join(hub_dir, 'inaSpeechSegmenter')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    fpath = os.path.join(target_dir, model_fname)
    
    if os.path.exists(fpath):
        return fpath

    # Download
    url_base = dmodels[model_fname]
    url = url_base + model_fname
    print(f"Downloading {url} to {fpath}")
    torch.hub.download_url_to_file(url, fpath, progress=True)
    
    return fpath
