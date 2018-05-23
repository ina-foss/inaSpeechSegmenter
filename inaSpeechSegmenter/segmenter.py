#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2018 Ina (David Doukhan - http://www.ina.fr/)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os
import tempfile
from subprocess import Popen, PIPE
import numpy as np
import keras
from keras import backend as KB
from .thread_returning import ThreadReturning

from skimage.util import view_as_windows as vaw

os.environ['SIDEKIT'] = 'theano=false,libsvm=false'
from sidekit.frontend.io import read_wav
from sidekit.frontend.features import mfcc

from pyannote.algorithms.utils.viterbi import viterbi_decoding
from .viterbi_utils import log_trans_exp, pred2logemission


graph = KB.get_session().graph      # To prevent the issue of keras with tensorflow backend for async tasks


def _wav2feats(wavname):
    """
    """
    ext = os.path.splitext(wavname)[-1]
    assert ext.lower() == '.wav' or ext.lower() == '.wave'
    sig, read_framerate, sampwidth = read_wav(wavname)
    shp = sig.shape
    # wav should contain a single channel
    assert len(shp) == 1 or (len(shp) == 2 and shp[1] == 1)
    # wav sample rate should be 16000 Hz
    assert read_framerate == 16000
    sig *= (2**(15-sampwidth))
    _, loge, _, mspec = mfcc(sig.astype(np.float32), get_mspec=True)
    return mspec, loge


def _energy_activity(loge, ratio=0.03):
    threshold = np.mean(loge[np.isfinite(loge)]) + np.log(ratio)
    raw_activity = (loge > threshold)
    return viterbi_decoding(pred2logemission(raw_activity),
                            log_trans_exp(150, cost0=-5))


def _get_patches(mspec, w, step):
    h = mspec.shape[1]
    data = vaw(mspec, (w,h), step=step)
    data.shape = (len(data), w*h)
    data = (data - np.mean(data, axis=1).reshape((len(data), 1))) / np.std(data, axis=1).reshape((len(data), 1))
    lfill = [data[0,:].reshape(1, h*w)] * (w // (2 * step))
    rfill = [data[-1,:].reshape(1, h*w)] * (w // (2* step) - 1 + len(mspec) % 2)
    data = np.vstack(lfill + [data] + rfill )
    finite = np.all(np.isfinite(data), axis=1)
    data.shape = (len(data), w, h)
    return data, finite


def _speechzic(nn, patches, finite_patches, vad):
    ret = []
    for lab, start, stop in _binidx2seglist(vad):
        if lab == 0:
            # no energy
            ret.append(('NOACTIVITY', start, stop))
            continue
        #print(start, stop)
        rawpred = nn.predict(patches[start:stop, :])
        rawpred[finite_patches[start:stop] == False, :] = 0.5
        pred = viterbi_decoding(np.log(rawpred), log_trans_exp(150))
        for lab2, start2, stop2 in _binidx2seglist(pred):
            ret.append((['Speech', 'Music'][int(lab2)], start2+start, stop2+start))
    return ret


def _gender(nn, patches, finite_patches, speechzicseg):
    ret = []
    for lab, start, stop in speechzicseg:
        if lab in ['Music', 'NOACTIVITY']:
            # no energy
            ret.append((lab, start, stop))
            continue
        rawpred = nn.predict(patches[start:stop, :])
        rawpred[finite_patches[start:stop] == False, :] = 0.5
        pred = viterbi_decoding(np.log(rawpred), log_trans_exp(80))
        for lab2, start2, stop2 in _binidx2seglist(pred):
            ret.append((['Female', 'Male'][int(lab2)], start2+start, stop2+start))
    return ret

def _binidx2seglist(binidx):
    curlabel = None
    bseg = -1
    ret = []
    for i, e in enumerate(binidx):
        if e != curlabel:
            if curlabel is not None:
                ret.append((curlabel, bseg, i))
            curlabel = e
            bseg = i
    ret.append((curlabel, bseg, i))
    return ret


def audio2features(ffmpeg, wavname, tmpwav, hasrf):
    """
    Convert wavname to computable feature for audio segmentation. hasrf is a boolean,
    set it to true if the file has the right format, i.e. WAVE 16K mono, else to false.
    """
    if not hasrf:
        # Convert the file to Wav 16K mono format
        args = [ffmpeg, '-y', '-i', wavname, '-ar', '16000', '-ac', '1', tmpwav]
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        output, error = p.communicate()
        assert p.returncode == 0, error

    # Get Mel Power Spectrogram and Energy
    mspec, loge = _wav2feats(wavname)
    # perform energy-based activity detection
    vad = _energy_activity(loge)[::2]

    data21, finite = _get_patches(mspec[:, :21], 68, 2)
    assert len(data21) == len(vad), (len(data21), len(vad))
    assert len(finite) == len(data21), (len(data21), len(finite))
    data, _ = _get_patches(mspec, 68, 2)
    return data, data21, finite, vad


def inference(sznn, gendernn, data, data21, finite, vad):
    """
    Make inference and produce segmentation informations based on
    the sznn music recognition network and gendernn gender recognition network.
    """
    global graph
    with graph.as_default():  # To prevent the issue of keras with tensorflow backend for async tasks
        # perform speech/music segmentation using only 21 MFC coefficients
        szseg = _speechzic(sznn, data21, finite, vad)
        genderseg = _gender(gendernn, data, finite, szseg)
        return [(lab, start * .02, stop * .02) for lab, start, stop in genderseg]



class Segmenter:
    def __init__(self):
        """
        Load neural network models
        """
        p = os.path.dirname(os.path.realpath(__file__)) + '/'
        self.sznn = keras.models.load_model(p + 'keras_speech_music_cnn.hdf5')
        self.gendernn = keras.models.load_model(p + 'keras_male_female_cnn.hdf5')
        self.i = 0


    def segmentwav(self, mediaList, ffmpeg='ffmpeg', tmpdir=None, hasrf=False): # hasrf -> has the right format. # TODO: Allow a boolean matrix
        """
        do segmentation on any kind on media file, including urls. medianame must be a list.
        Quicker if input corresponding to wav file(s) sampled at 16000Hz
        with a single channel. If so, hasrf must be set to True.
        """
        alles = [os.path.splitext(os.path.basename(e)) for e in mediaList]
        base = [alles[i][0] for i in range(len(alles))]
        # ext = [alles[i][1] for i in range(len(alles))]

        with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:
            tmpwav = ['%s/%s.wav' % (tmpdirname, elem) for elem in base]
            list_of_data = list()
            returns_from_cpu, returns_from_gpu = None, None
            # We run the first CPU task
            cpu_thread = ThreadReturning(target=audio2features, args=(ffmpeg, mediaList[0], tmpwav[0], hasrf))
            cpu_thread.start()
            returns_from_cpu = cpu_thread.join()
            for media_name, tmp_wav in zip(mediaList[1:], tmpwav[1:]):
                data, data21, finite, vad = returns_from_cpu
                # While running the n th GPU task, we run the n+1 th CPU task
                gpu_thread = ThreadReturning(target=inference, args=(self.sznn, self.gendernn, data, data21, finite, vad))
                cpu_thread = ThreadReturning(target=audio2features, args=(ffmpeg, media_name, tmp_wav, hasrf))
                cpu_thread.start(), gpu_thread.start()
                returns_from_gpu, returns_from_cpu = gpu_thread.join(), cpu_thread.join()
                # Storing in .csv file or printed (if asked)
                self.seg2csv(returns_from_gpu)
                list_of_data.append(returns_from_gpu)
            # Then running the last GPU task
            data, data21, finite, vad = returns_from_cpu
            gpu_thread = ThreadReturning(target=inference, args=(self.sznn, self.gendernn, data, data21, finite, vad))
            gpu_thread.start()
            returns_from_gpu = gpu_thread.join()
            list_of_data.append(returns_from_gpu)
            self.seg2csv(returns_from_gpu)
            KB.clear_session()  # End Keras issues
            return list_of_data


    def seg2csv(self, seg):
        print("Processing file n°{} ...".format(self.i))
        if self.fout is None:
            for lab, beg, end in seg:
                print('%s\t%f\t%f' % (lab, beg, end))
        else:
            with open(self.fout[self.i], 'wt') as fid:
                for lab, beg, end in seg:
                    fid.write('%s\t%f\t%f\n' % (lab, beg, end))
            self.i += 1
        print("... done.")


    def __call__(self, medianame, ffmpeg='ffmpeg', tmpdir=None, hasrf=False, fout=None):
        """
        do segmentation on any kind on media file, including urls
        """
        self.fout = fout
        self.segmentwav(medianame, ffmpeg, tmpdir, hasrf)
        print("Segmentation done.")






def to_parse(input_files):
    """
    Return an explicit list of all input files to segment (path and url).
    Parse .txt file if found, where each line is a file to segment.
    """
    if any(file.endswith(".txt") for file in input_files):
        files = list()
        for e in input_files:
            if e.endswith(".txt"):
                with open(e) as src_fd :
                    for line in src_fd:
                        a = line.strip(" \t\n\r").rstrip('/')  # Removing first and last spaces, returns, or slashes.
                        if len(a) > 0:
                            files.append(a)
            else:
                files.append(e)
    else:
        files = list(input_files)
    return files

