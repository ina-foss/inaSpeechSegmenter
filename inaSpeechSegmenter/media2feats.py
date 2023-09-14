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
import soundfile as sf
import shutil
import types
import warnings
import numpy as np

from .sidekit_mfcc import mfcc
from .vbx_melbands import vbx_melbands

def media2sig16kmono(medianame, tmpdir=None, start_sec=None, stop_sec=None, ffmpeg='ffmpeg', dtype='float64'):
    """
    Convert media to temp wav 16k mono and return signal
    """

    base, _ = os.path.splitext(os.path.basename(medianame))

    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:
        # build ffmpeg command line
        tmpwav = tmpdirname + '/' + base + '.wav'
        args = [ffmpeg, '-y', '-i', medianame, '-ar', '16000', '-ac', '1']
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
        
        # error management
        if p.returncode != 0:
            if 'No such file or directory' in str(error):
                raise FileNotFoundError(medianame)
            elif 'HTTP error 404 Not Found' in str(error):
                raise FileNotFoundError('Server returned 404 Not Found error for url ' + medianame)
        assert p.returncode == 0, error

        # Get Mel Power Spectrogram and Energy
        sig, sr = sf.read(tmpwav, dtype=dtype)
        assert sr == 16000
        return sig
    
    
class CpuFeatExtractor:
    """
    Extract all CPU features from audio or video document
    this includes : 
        * download from a remote location
        * transcoding to wav 16k
        * mel bands extraction with sidekits and/or librosa
    Depending on use-case, this can be used asynchronously in parallel with
    other GPU-based processings
    """
    def __init__(self, sdkmel, vbxmel, ffmpeg, tmpdir):
        self.sdkmel = sdkmel
        self.vbxmel = vbxmel

        # test ffmpeg installation
        if shutil.which(ffmpeg) is None:
            raise (Exception("""ffmpeg program not found"""))
        self.ffmpeg = ffmpeg
        self.tmpdir = tmpdir
        
    def __call__(self, medianame, start_sec, stop_sec):
        
        # result container
        ret = types.SimpleNamespace()

        # start_sec seconds will be skipped from decoding and feature xtraction
        # this offset should be kept
        if start_sec is None:
            ret.start_sec = 0
        else:
            ret.start_sec = start_sec
        
        # transcoding to wav16k
        sig = media2sig16kmono(medianame, self.tmpdir, start_sec, stop_sec, ffmpeg=self.ffmpeg, dtype="float64")
        
        # signal duration
        ret.duration = len(sig) / 16000.
        
        # sidekits mel bands xtraction (default)
        if self.sdkmel:
            with warnings.catch_warnings():
                # ignore warnings resulting from empty signals parts
                warnings.filterwarnings('ignore', message='divide by zero encountered in log', category=RuntimeWarning)
                _, loge, _, mspec = mfcc(sig.astype(np.float32), get_mspec=True)
        
            # Management of short duration segments
            difflen = 0
            if len(loge) < 68:
                difflen = 68 - len(loge)
                warnings.warn(
                    "media %s duration is short. Robust results require length of at least 720 milliseconds" % medianame)
                mspec = np.concatenate((mspec, np.ones((difflen, 24)) * np.min(mspec)))
                
            ret.mspec, ret.loge, ret.difflen = (mspec, loge, difflen)
        
        # librosa mel bands xtraction
        if self.vbxmel:
            ret.mspec_vbx = vbx_melbands(sig)
        
        return ret