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


class DnnSegmenter:
    """
    DnnSegmenter is an abstract class allowing to perform Dnn-based
    segmentation using Keras serialized models using 24 mel spectrogram
    features obtained with SIDEKIT framework.

    Child classes MUST define the following class attributes:
    * nmel: the number of mel bands to used (max: 24)
    * viterbi_arg: the argument to be used with viterbi post-processing
    * model_fname: the filename of the serialized keras model to be used
        the model should be stored in the current directory
    * inlabel: only segments with label name inlabel will be analyzed.
        other labels will stay unchanged
    * outlabels: the labels associated the output of neural network models
    """
    def __init__(self):
        # load the DNN model
        p = os.path.dirname(os.path.realpath(__file__)) + '/'
        self.nn = keras.models.load_model(p + self.model_fname, compile=False)        
    
    def __call__(self, mspec, lseg, difflen = 0):
        """
        *** input
        * mspec: mel spectrogram
        * lseg: list of tuples (label, start, stop) corresponding to previous segmentations
        * difflen: 0 if the original length of the mel spectrogram is >= 68
                otherwise it is set to 68 - length(mspec)
        *** output
        a list of adjacent tuples (label, start, stop)
        """

        if self.nmel < 24:
            mspec = mspec[:, :self.nmel].copy()
        
        patches, finite = _get_patches(mspec, 68, 2)
        if difflen > 0:
            patches = patches[:-int(difflen / 2), :, :]
            finite = finite[:-int(difflen / 2)]
            
        assert len(finite) == len(patches), (len(patches), len(finite))
            
        batch = []
        for lab, start, stop in lseg:
            if lab == self.inlabel:
                batch.append(patches[start:stop, :])

        if len(batch) > 0:
            batch = np.concatenate(batch)
            rawpred = self.nn.predict(batch)

        ret = []
        for lab, start, stop in lseg:
            if lab != self.inlabel:
                ret.append((lab, start, stop))
                continue

            l = stop - start
            r = rawpred[:l] 
            rawpred = rawpred[l:]
            r[finite[start:stop] == False, :] = 0.5
            pred = viterbi_decoding(np.log(r), diag_trans_exp(self.viterbi_arg, len(self.outlabels)))
            for lab2, start2, stop2 in _binidx2seglist(pred):
                ret.append((self.outlabels[int(lab2)], start2+start, stop2+start))            
        return ret


class SpeechMusic(DnnSegmenter):
    # Voice activity detection: requires energetic activity detection
    outlabels = ('speech', 'music')
    model_fname = 'keras_speech_music_cnn.hdf5'
    inlabel = 'energy'
    nmel = 21
    viterbi_arg = 150

class SpeechMusicNoise(DnnSegmenter):
    # Voice activity detection: requires energetic activity detection
    outlabels = ('speech', 'music', 'noise')
    model_fname = 'keras_speech_music_noise_cnn.hdf5'
    inlabel = 'energy'
    nmel = 21
    viterbi_arg = 80
    
class Gender(DnnSegmenter):
    # Gender Segmentation, requires voice activity detection
    outlabels = ('female', 'male')
    model_fname = 'keras_male_female_cnn.hdf5'
    inlabel = 'speech'
    nmel = 24
    viterbi_arg = 80


class Segmenter:
    def __init__(self, vad_engine='smn', detect_gender=True, ffmpeg='ffmpeg'):
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
            self.gender = Gender()
            

    def segmentwav(self, wavname, start_sec):
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
        lseg = []
        for lab, start, stop in _binidx2seglist(_energy_activity(loge)[::2]):
            if lab == 0:
                lab = 'noEnergy'
            else:
                lab = 'energy'
            lseg.append((lab, start, stop))

        # perform voice activity detection
        lseg = self.vad(mspec, lseg, difflen)

        # perform gender segmentation on speech segments
        if self.detect_gender:
            lseg = self.gender(mspec, lseg, difflen)

        # TODO: OFFSET MANAGEMENT
        return [(lab, start_sec + start * .02, start_sec + stop * .02) for lab, start, stop in lseg]


    def __call__(self, medianame, tmpdir=None, start_sec=None, stop_sec=None):
        """
        Return segmentation of a given file
                * convert file to wav 16k mono with ffmpeg
                * call NN segmentation procedures
        * media_name: path to the media to be processed (including remote url)
                may include any format supported by ffmpeg
        * tmpdir: allow to define a custom path for storing temporary files
                fast read/write HD are a good choice
        * start_sec (seconds): sound stream before start_sec won't be processed
        * stop_sec (seconds): sound stream after stop_sec won't be processed
        """
        
        base, _ = os.path.splitext(os.path.basename(medianame))

        with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:
            # build ffmpeg command line
            tmpwav = tmpdirname + '/' + base + '.wav'
            args = [self.ffmpeg, '-y', '-i', medianame, '-ar', '16000', '-ac', '1']
            if start_sec is None:
                start_sec = 0
            else:
                args += ['-ss', '%f' % start_sec]
                
            if stop_sec is not None:
                args += ['-to', '%f' % stop_sec]
            args += [tmpwav]

            # launch ffmpeg
            p = Popen(args, stdout=PIPE, stderr=PIPE)
            output, error = p.communicate()
            assert p.returncode == 0, error
            
            # do segmentation
            return self.segmentwav(tmpwav, start_sec)

def seg2csv(lseg, fout=None):
    df = pd.DataFrame.from_records(lseg, columns=['labels', 'start', 'stop'])
    df.to_csv(fout, sep='\t', index=False)

