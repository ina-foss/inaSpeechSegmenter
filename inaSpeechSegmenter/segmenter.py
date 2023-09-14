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
import sys
import numpy as np



# Tells GPU not to use all available memory
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
from tensorflow import keras


from .thread_returning import ThreadReturning

import time
import random
import gc

from skimage.util import view_as_windows as vaw

from pyannote.algorithms.utils.viterbi import viterbi_decoding
from .viterbi_utils import pred2logemission, diag_trans_exp, log_trans_exp
from .remote_utils import get_remote

from .vbxsegmenter import VBxSegmenter
from .utils import binidx2seglist

from .export_funcs import seg2csv, seg2textgrid

from .media2feats import CpuFeatExtractor



def _energy_activity(loge, ratio):
    threshold = np.mean(loge[np.isfinite(loge)]) + np.log(ratio)
    raw_activity = (loge > threshold)
    return viterbi_decoding(pred2logemission(raw_activity),
                            log_trans_exp(150, cost0=-5))


def _get_patches(mspec, w, step):
    h = mspec.shape[1]
    data = vaw(mspec, (w, h), step=step)
    data.shape = (len(data), w * h)
    data = (data - np.mean(data, axis=1).reshape((len(data), 1))) / np.std(data, axis=1).reshape((len(data), 1))
    lfill = [data[0, :].reshape(1, h * w)] * (w // (2 * step))
    rfill = [data[-1, :].reshape(1, h * w)] * (w // (2 * step) - 1 + len(mspec) % 2)
    data = np.vstack(lfill + [data] + rfill)
    finite = np.all(np.isfinite(data), axis=1)
    data.shape = (len(data), w, h)
    return data, finite


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
    def __init__(self, batch_size):
        # load the DNN model

        model_path = get_remote(self.model_fname)

        self.nn = keras.models.load_model(model_path, compile=False)
        self.nn.run_eagerly = False
        self.batch_size = batch_size

    def __call__(self, feats, lseg):
        """
        *** input
        * mspec: mel spectrogram
        * lseg: list of tuples (label, start, stop) corresponding to previous segmentations
        * difflen: 0 if the original length of the mel spectrogram is >= 68
                otherwise it is set to 68 - length(mspec)
        *** output
        a list of adjacent tuples (label, start, stop)
        """

        mspec = feats.mspec
        difflen = feats.difflen

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
            rawpred = self.nn.predict(batch, batch_size=self.batch_size, verbose=2)
        gc.collect()

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
            for lab2, start2, stop2 in binidx2seglist(pred):
                ret.append((self.outlabels[int(lab2)], start2 + start, stop2 + start))
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
    def __init__(self, vad_engine='smn', gender_engine='ic18', ffmpeg='ffmpeg',
                 batch_size=32, energy_ratio=0.03, tmpdir=None):
        """
        Load neural network models
        
        Input:

        'vad_engine' can be 'sm' (speech/music) or 'smn' (speech/music/noise)
                'sm' was used in the results presented in ICASSP 2017 paper
                        and in MIREX 2018 challenge submission
                'smn' has been implemented more recently and has not been evaluated in papers
        
        'gender_engine': speaker gender segmentation engine (default ic18)
            if None, speech excerpts are return labelled as 'speech' (faster)
            Otherwise speech excerpts are splitted into 'male' and 'female' segments
            if 'ic18' is chosen (default): CNN model presentend at ICASSP 2018 is used
            if 'is23' is chosen : X-vector model presented at Interspeech 2023 is used (slower)

        'batch_size' : large values of batch_size (ex: 1024) allow faster processing times.
                They also require more memory on the GPU.
                default value (32) is slow, but works on any hardware
        'tmpdir' : allow to define a custom path for storing temporary files
                fast read/write HD are a good choice
        """

        extract_vbx_mels = False

        # set energic ratio for 1st VAD
        self.energy_ratio = energy_ratio

        # select speech/music or speech/music/noise voice activity detection engine
        assert vad_engine in ['sm', 'smn']
        if vad_engine == 'sm':
            self.vad = SpeechMusic(batch_size)
        elif vad_engine == 'smn':
            self.vad = SpeechMusicNoise(batch_size)



        # load gender detection NN if required
        if gender_engine is None or gender_engine.lower() == 'none':
            self.gender = None
        elif gender_engine == 'ic18':
            self.gender = Gender(batch_size)
        elif gender_engine == 'is23':
            # TODO : batch_size management
            self.gender = VBxSegmenter()
            extract_vbx_mels = True
        else:
            raise ValueError('Invalid value "%s" provided to gender_engine. Allowed values are "ic18", "is23" or None' % gender_engine)
            
        self.feat_extractor = CpuFeatExtractor(True, extract_vbx_mels, ffmpeg, tmpdir)

    def segment_feats(self, feats):
        # TODO : mspec_vbx should not be argument... refactor ??
        """
        do segmentation
        require input corresponding to wav file sampled at 16000Hz
        with a single channel
        """
        # perform energy-based activity detection
        lseg = []
        for lab, start, stop in binidx2seglist(_energy_activity(feats.loge, self.energy_ratio)[::2]):
            if lab == 0:
                lab = 'noEnergy'
            else:
                lab = 'energy'
            lseg.append((lab, start, stop))

        # perform voice activity detection
        lseg = self.vad(feats, lseg)

        # perform gender segmentation on speech segments using the DnnSegmenter Class/ VBxSegmenter Class
        #TODO : all classes should have same signature : 
        # single feature instance containing all required data)
        # difflen argument
        if self.gender is not None:
            lseg = self.gender(feats, lseg)

        # TODO : 0.2 strange for vbx based
        start_sec = feats.start_sec
        return [(lab, start_sec + start * .02, start_sec + stop * .02) for lab, start, stop in lseg]


    def __call__(self, medianame, start_sec=None, stop_sec=None):
        """
        Return segmentation of a given file
                * convert file to wav 16k mono with ffmpeg
                * call NN segmentation procedures
        * media_name: path to the media to be processed (including remote url)
                may include any format supported by ffmpeg
        * start_sec (seconds): sound stream before start_sec won't be processed
        * stop_sec (seconds): sound stream after stop_sec won't be processed
        """

        # extract CPU features
        feats = self.feat_extractor(medianame, start_sec, stop_sec)
        # do GPU processings from CPU features
        return self.segment_feats(feats)
        
    
    def batch_process(self, linput, loutput, verbose=False, skipifexist=False, nbtry=1, trydelay=2.,
                      output_format='csv'):

        if verbose:
            print('batch_processing %d files' % len(linput))

        if output_format == 'csv':
            fexport = seg2csv
        elif output_format == 'textgrid':
            fexport = seg2textgrid
        else:
            raise NotImplementedError()

        t_batch_start = time.time()

        lmsg = []
        fg = featGenerator(self.feat_extractor, linput.copy(), loutput.copy(), skipifexist, nbtry, trydelay)
        i = 0
        for feats, msg in fg:
            lmsg += msg
            i += len(msg)
            if verbose:
                print('%d/%d' % (i, len(linput)), msg)
            if feats is None:
                break
            #mspec, loge, difflen = feats
            # if verbose == True:
            #    print(i, linput[i], loutput[i])
            b = time.time()
            lseg = self.segment_feats(feats)
            fexport(lseg, loutput[len(lmsg) - 1])
            lmsg[-1] = (lmsg[-1][0], lmsg[-1][1], 'ok ' + str(time.time() - b))

        t_batch_dur = time.time() - t_batch_start
        nb_processed = len([e for e in lmsg if e[1] == 0])
        if nb_processed > 0:
            avg = t_batch_dur / nb_processed
        else:
            avg = -1
        return t_batch_dur, nb_processed, avg, lmsg


def medialist2feats(extractor, lin, lout, skipifexist, nbtry, trydelay):
    """
    To be used when processing batches
    if resulting file exists, it is skipped
    in case of remote files, access is tried nbtry times
    """
    ret = None
    msg = []
    while ret is None and len(lin) > 0:
        src = lin.pop(0)
        dst = lout.pop(0)

        # if file exists: skipp
        if skipifexist and os.path.exists(dst):
            msg.append((dst, 1, 'already exists'))
            continue

        # create storing directory if required
        dname = os.path.dirname(dst)
        if not os.path.isdir(dname):
            os.makedirs(dname)

        itry = 0
        while ret is None and itry < nbtry:
            try:
                ret = extractor(src, None, None)
            except:
                itry += 1
                errmsg = sys.exc_info()[0]
                if itry != nbtry:
                    time.sleep(random.random() * trydelay)
        if ret is None:
            msg.append((dst, 2, 'error: ' + str(errmsg)))
        else:
            msg.append((dst, 0, 'ok'))

    return ret, msg


def featGenerator(extractor, ilist, olist, skipifexist=False, nbtry=1, trydelay=2.):
    thread = ThreadReturning(target=medialist2feats, args=[extractor, ilist, olist, skipifexist, nbtry, trydelay])
    thread.start()
    while True:
        ret, msg = thread.join()
        if len(ilist) == 0:
            break
        thread = ThreadReturning(target=medialist2feats,
                                 args=[extractor, ilist, olist, skipifexist, nbtry, trydelay])
        thread.start()
        yield ret, msg
    yield ret, msg
