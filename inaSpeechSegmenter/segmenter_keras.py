
import os
import sys
import numpy as np
import shutil
import time
import random
import gc
import warnings

# Import TensorFlow/Keras for the legacy backend
from tensorflow import keras
from tensorflow.keras.utils import get_file

from .thread_returning import ThreadReturning
from skimage.util import view_as_windows as vaw

from .pyannote_viterbi import viterbi_decoding
from .viterbi_utils import pred2logemission, diag_trans_exp, log_trans_exp
from .remote_utils import get_remote

from .io import media2sig16kmono
from .sidekit_mfcc import mfcc
from .export_funcs import seg2csv, seg2textgrid

# Re-use helper functions from segmenter.py to avoid code duplication if possible,
# or redefine them if they were modified.
# _media2feats, _energy_activity, _get_patches, _binidx2seglist were not modified significantly logic-wise,
# but to be safe and self-contained, I'll reproduce them here or import them if they are identical.
# They seem identical in the PyTorch version (except imports).

from .segmenter import _media2feats, _energy_activity, _get_patches, _binidx2seglist, featGenerator, medialist2feats

class DnnSegmenter:
    """
    (Legacy Keras Version)
    DnnSegmenter is an abstract class allowing to perform Dnn-based
    segmentation using Keras serialized models using 24 mel spectrogram
    features obtained with SIDEKIT framework.
    """
    def __init__(self, batch_size):
        # load the DNN model
        model_path = get_remote(self.model_fname)
        self.nn = keras.models.load_model(model_path, compile=False)
        self.nn.run_eagerly = False
        self.batch_size = batch_size

    def __call__(self, mspec, lseg, difflen = 0):
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
            batch = np.expand_dims(np.concatenate(batch), 3)
            rawpred = self.nn.predict(batch, batch_size=self.batch_size, verbose=2)
        gc.collect()
            
        ret = []
        # Reconstruct segmentation
        # We need to distribute rawpred back to the segments
        # Note: The original code logic for reconstruction:
        
        # original code:
        # for lab, start, stop in lseg:
        #     if lab != self.inlabel:
        #         ret.append((lab, start, stop))
        #         continue
        #     l = stop - start
        #     r = rawpred[:l] 
        #     rawpred = rawpred[l:]
        #     ...

        # Yes, rawpred is consumed
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
    def __init__(self, vad_engine='smn', detect_gender=True, ffmpeg='ffmpeg', batch_size=32, energy_ratio=0.03):
        """
        Load neural network models (Legacy Keras Implementation)
        """      

        if ffmpeg is not None:
            if shutil.which(ffmpeg) is None:
                raise(Exception("""ffmpeg program not found"""))
        self.ffmpeg = ffmpeg
        self.energy_ratio = energy_ratio

        assert vad_engine in ['sm', 'smn']
        if vad_engine == 'sm':
            self.vad = SpeechMusic(batch_size)
        elif vad_engine == 'smn':
            self.vad = SpeechMusicNoise(batch_size)

        assert detect_gender in [True, False]
        self.detect_gender = detect_gender
        if detect_gender:
            self.gender = Gender(batch_size)


    def segment_feats(self, mspec, loge, difflen, start_sec):
        lseg = []
        for lab, start, stop in _binidx2seglist(_energy_activity(loge, self.energy_ratio)[::2]):
            if lab == 0:
                lab = 'noEnergy'
            else:
                lab = 'energy'
            lseg.append((lab, start, stop))

        lseg = self.vad(mspec, lseg, difflen)

        if self.detect_gender:
            lseg = self.gender(mspec, lseg, difflen)

        return [(lab, start_sec + start * .02, start_sec + stop * .02) for lab, start, stop in lseg]


    def __call__(self, medianame, start_sec=None, stop_sec=None):
        mspec, loge, difflen = _media2feats(medianame, start_sec, stop_sec, self.ffmpeg)
        if start_sec is None:
            start_sec = 0
        return self.segment_feats(mspec, loge, difflen, start_sec)

    
    def batch_process(self, linput, loutput, verbose=False, skipifexist=False, nbtry=1, trydelay=2., output_format='csv'):
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
        fg = featGenerator(linput.copy(), loutput.copy(), self.ffmpeg, skipifexist, nbtry, trydelay)
        i = 0
        for feats, msg in fg:
            lmsg += msg
            i += len(msg)
            if verbose:
                print('%d/%d' % (i, len(linput)), msg)
            if feats is None:
                break
            mspec, loge, difflen = feats
            b = time.time()
            lseg = self.segment_feats(mspec, loge, difflen, 0)
            fexport(lseg, loutput[len(lmsg) -1])
            lmsg[-1] = (lmsg[-1][0], lmsg[-1][1], 'ok ' + str(time.time() -b))

        t_batch_dur = time.time() - t_batch_start
        nb_processed = len([e for e in lmsg if e[1] == 0])
        if nb_processed > 0:
            avg = t_batch_dur / nb_processed
        else:
            avg = -1
        return t_batch_dur, nb_processed, avg, lmsg
