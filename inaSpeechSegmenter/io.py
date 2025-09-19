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
from tempfile import TemporaryFile
import subprocess
import soundfile as sf


def media2sig16kmono(medianame, start_sec=None, stop_sec=None, ffmpeg='ffmpeg', dtype='float64'):
    """
    Convert media to temp wav 16k mono and return signal
    """

    if ffmpeg is None:
        if start_sec is not None or stop_sec is not None:
            raise NotImplementedError(
                f'start_sec={start_sec} and stop_sec={stop_sec} cannot be set ' \
                f' when running inaSpeechSegmenter without ffmpeg. Please cut '\
                f'down your audio files beforehand or use ffmpeg.'
            )
        if medianame.startswith('http://') or medianame.startswith('https://'):
            raise NotImplementedError(
                f'Without ffmpeg you cannot process media content on http ' \
                f'servers. You need to download your audio files beforehand ' \
                f'or use ffmpeg. You gave medianame={medianame}.'
            )

        sig, sr = sf.read(medianame, dtype=dtype)
        assert sr == 16_000, \
            f'Without ffmpeg, inaSpeechSegmenter can only take files sampled ' \
            f'at 16000 Hz. The file {medianame} is sampled at {sr} Hz.'
        return sig

    base, _ = os.path.splitext(os.path.basename(medianame))


    # build ffmpeg command
    cmd = [ffmpeg, '-i', medianame, '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1']
    if start_sec is None:
        start_sec = 0
    else:
        cmd += ['-ss', '%f' % start_sec]
    if stop_sec is not None:
        cmd += ['-to', '%f' % stop_sec]
    cmd += ['pipe:1']

    with TemporaryFile() as out, TemporaryFile() as err:
        ret = subprocess.run(cmd, stdout=out, stderr=err)
        if ret.returncode != 0:
            err.seek(0)
            msg = err.read()
            raise Exception(msg)
        out.seek(0)
        wav_data, fs = sf.read(out, dtype=dtype)
    assert(fs == 16000)
    return wav_data
