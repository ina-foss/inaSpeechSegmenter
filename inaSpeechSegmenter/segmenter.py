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
import shutil
import pandas as pd
import warnings

from skimage.util import view_as_windows as vaw

os.environ['SIDEKIT'] = 'theano=false,libsvm=false'
from sidekit.frontend.io import read_wav
from sidekit.frontend.features import mfcc

from pyannote.algorithms.utils.viterbi import viterbi_decoding
from .viterbi_utils import pred2logemission, diag_trans_exp, log_trans_exp


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
    assert sampwidth == 2
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


def _gender(nn, patches, finite_patches, speechzicseg):
    ret = []
    for lab, start, stop in speechzicseg:
        if lab != 'speech':#in ['Music', 'NOACTIVITY']:
            # no energy
            ret.append((lab, start, stop))
            continue
        rawpred = nn.predict(patches[start:stop, :])
        rawpred[finite_patches[start:stop] == False, :] = 0.5
        pred = viterbi_decoding(np.log(rawpred), log_trans_exp(80))
        for lab2, start2, stop2 in _binidx2seglist(pred):
            ret.append((['female', 'male'][int(lab2)], start2+start, stop2+start))
    return ret

def _binidx2seglist(binidx):
    """
    ss._binidx2seglist((['f'] * 5) + (['bbb'] * 10) + ['v'] * 5)
    Out: [('f', 0, 5), ('bbb', 5, 15), ('v', 15, 20)]
    
    #TODO: is there a pandas alternative??
    """
    curlabel = None
    bseg = -1
    ret = []
    for i, e in enumerate(binidx):
        if e != curlabel:
            if curlabel is not None:
                ret.append((curlabel, bseg, i))
            curlabel = e
            bseg = i
    ret.append((curlabel, bseg, i + 1))
    return ret


class Vad:
    """
    Base class to be used between signal activity and gender segmentation.
    It splits signal labelled as active into segments labelled as speech
    and/or other types of signals (music, noise, ...)
    The input of this modules correspond to the 21 first bands of the mel
    spectrogram
    """
    def __init__(self):
        p = os.path.dirname(os.path.realpath(__file__)) + '/'
        self.nn = keras.models.load_model(p + self.model_fname, compile=False)        
    
    def __call__(self, mspec, lseg, difflen = 0):
        """
        * input
        mspec: 21 bands mel spectrogram
        difflen: 0 if the original length of the mel spectrogram is >= 68
                otherwise it is set to 68 - length(mspec)
        """
        patches, finite = _get_patches(mspec[:, :21].copy(), 68, 2)
        if difflen > 0:
            patches = patches[:-int(difflen / 2), :, :]
            finite = finite[:-int(difflen / 2)]
            
        assert len(finite) == len(patches), (len(patches), len(finite))
            
        ret = []
        for lab, start, stop in lseg:
            if lab != self.inlabel:
                ret.append((lab, start, stop))
                continue

            rawpred = self.nn.predict(patches[start:stop, :])
            rawpred[finite[start:stop] == False, :] = 0.5

            # specific code bellow
            pred = viterbi_decoding(np.log(rawpred), diag_trans_exp(150, len(self.outlabels)))
            for lab2, start2, stop2 in _binidx2seglist(pred):
                ret.append((self.outlabels[int(lab2)], start2+start, stop2+start))            
        return ret


class SpeechMusic(Vad):
    outlabels = ('speech', 'music')
    model_fname = 'keras_speech_music_cnn.hdf5'
    inlabel = 'energy'

class SpeechMusicNoise(Vad):
    outlabels = ('speech', 'music', 'noise')
    model_fname = 'keras_speech_music_noise_cnn.hdf5'
    inlabel = 'energy'
    
class Gender(Vad):
    outlabels = ('female', 'male')
    model_fname = 'keras_male_female_cnn.hdf5'
    inlabel = 'speech'

class Segmenter:
    def __init__(self, vad_engine='sm', detect_gender=True, ffmpeg='ffmpeg'):
        """
        Load neural network models
        
        Input:

        'vad_engine' can be 'sm' (speech/music) or 'smn' (speech/music/noise)
                'sm' was used in the results presented in ICASSP 2017 paper
                        and in MIREX 2018 challenge submission
                'smn' has been implemented more recently and has not been evaluated in papers
        
        'detect_gender': if False, speech excerpts are return labelled as 'speech'
                if True, speech excerpts are splitted into 'male' and 'female' segments
        """      

        # test ffmpeg installation
        if shutil.which(ffmpeg) is None:
            raise(Exception("""ffmpeg program not found"""))
        self.ffmpeg = ffmpeg

        # select speech/music or speech/music/noise voice activity detection engine
        assert vad_engine in ['sm', 'smn']
        if vad_engine == 'sm':
            self.vad = SpeechMusic()
        elif vad_engine == 'smn':
            self.vad = SpeechMusicNoise()

        # load gender detection NN if required
        assert detect_gender in [True, False]
        self.detect_gender = detect_gender
        if detect_gender:
            p = os.path.dirname(os.path.realpath(__file__)) + '/'
            self.gendernn = keras.models.load_model(p + 'keras_male_female_cnn.hdf5', compile=False)    

    def segmentwav(self, wavname):
        """
        do segmentation
        require input corresponding to wav file sampled at 16000Hz
        with a single channel
        """
        # Get Mel Power Spectrogram and Energy
        mspec, loge = _wav2feats(wavname)

        # Management of short duration segments
        difflen = 0
        if len(loge) < 68:
            difflen = 68 - len(loge)
            warnings.warn("media %s duration is short. Robust results require length of at least 720 milliseconds" %wavname)
            mspec = np.concatenate((mspec, np.ones((difflen, 24)) * np.min(mspec)))
            #loge = np.concatenate((loge, np.ones(difflen) * np.min(mspec)))


        # perform energy-based activity detection
        ead = []
        for lab, start, stop in _binidx2seglist(_energy_activity(loge)[::2]):
            if lab == 0:
                lab = 'noEnergy'
            else:
                lab = 'energy'
            ead.append((lab, start, stop))

        # perform voice activity detection
        speech_seg = self.vad(mspec, ead, difflen)
        if not self.detect_gender:
            # TODO: OFFSET MANAGEMENT
            return [(lab, start * .02, stop * .02) for lab, start, stop in speech_seg]

        # perform gender segmentation on 'speech' labelled segments
        data, finite = _get_patches(mspec, 68, 2)
        if difflen > 0:
            data = data[:-int(difflen/2), :, :]
            finite = finite[:-int(difflen/2)]        
        genderseg = _gender(self.gendernn, data, finite, speech_seg)
        # TODO: OFFSET MANAGEMENT
        return [(lab, start * .02, stop * .02) for lab, start, stop in genderseg]

    def __call__(self, medianame, tmpdir=None):
        """
        do segmentation on any kind on media file, including urls
        slower than segmentwav method
        """
        
        base, _ = os.path.splitext(os.path.basename(medianame))

        with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:
            tmpwav = tmpdirname + '/' + base + '.wav'
            args = [self.ffmpeg, '-y', '-i', medianame, '-ar', '16000', '-ac', '1', tmpwav]
            p = Popen(args, stdout=PIPE, stderr=PIPE)
            output, error = p.communicate()
            assert p.returncode == 0, error
            return self.segmentwav(tmpwav)

def seg2csv(lseg, fout=None):
    df = pd.DataFrame.from_records(lseg, columns=['labels', 'start', 'stop'])
    df.to_csv(fout, sep='\t', index=False)

