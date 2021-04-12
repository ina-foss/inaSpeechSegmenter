
# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# The following code has been copy-pasted from SIDEKIT source files:
# frontend/features.py frontend/io.py frontend/vad.py
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2021 Anthony Larcher and Sylvain Meignier

:mod:`frontend` provides methods to process an audio signal in order to extract
useful parameters for speaker verification.
"""


import numpy
import soundfile
import scipy
from scipy.fftpack.realtransforms import dct


__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2021 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'




wav_flag = "float32"    # Could be "int16"
PARAM_TYPE = numpy.float32


def read_wav(input_file_name):
    """
    :param input_file_name:
    :return:
    """
    #with wave.open(input_file_name, "r") as wfh:
    #    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wfh.getparams()
    #    raw = wfh.readframes(nframes * nchannels)
    #    out = struct.unpack_from("%dh" % nframes * nchannels, raw)
    #    sig = numpy.reshape(numpy.array(out), (-1, nchannels)).squeeze()
    #    return sig.astype(numpy.float32), framerate, sampwidth
    nfo = soundfile.info(input_file_name)
    sig, sample_rate = soundfile.read(input_file_name, dtype=wav_flag)
    sig = numpy.reshape(numpy.array(sig), (-1, nfo.channels)).squeeze()
    sig = sig.astype(numpy.float32)
    return sig, sample_rate, 4




def hz2mel(f, htk=True):
    """Convert an array of frequency in Hz into mel.
    
    :param f: frequency to convert
    
    :return: the equivalence on the mel scale.
    """
    if htk:
        return 2595 * numpy.log10(1 + f / 700.)
    else:
        f = numpy.array(f)

        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        f_0 = 0.
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = f < brkfrq

        z = numpy.zeros_like(f)
        # fill in parts separately
        z[linpts] = (f[linpts] - f_0) / f_sp
        z[~linpts] = brkpt + (numpy.log(f[~linpts] / brkfrq)) / numpy.log(logstep)

        if z.shape == (1,):
            return z[0]
        else:
            return z

def mel2hz(z, htk=True):
    """Convert an array of mel values in Hz.
    
    :param m: ndarray of frequencies to convert in Hz.
    
    :return: the equivalent values in Hertz.
    """
    if htk:
        return 700. * (10**(z / 2595.) - 1)
    else:
        z = numpy.array(z, dtype=float)
        f_0 = 0
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = (z < brkpt)

        f = numpy.zeros_like(z)

        # fill in parts separately
        f[linpts] = f_0 + f_sp * z[linpts]
        f[~linpts] = brkfrq * numpy.exp(numpy.log(logstep) * (z[~linpts] - brkpt))

        if f.shape == (1,):
            return f[0]
        else:
            return f



def trfbank(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt, midfreq=1000):
    """Compute triangular filterbank for cepstral coefficient computation.

    :param fs: sampling frequency of the original signal.
    :param nfft: number of points for the Fourier Transform
    :param lowfreq: lower limit of the frequency band filtered
    :param maxfreq: higher limit of the frequency band filtered
    :param nlinfilt: number of linear filters to use in low frequencies
    :param  nlogfilt: number of log-linear filters to use in high frequencies
    :param midfreq: frequency boundary between linear and log-linear filters

    :return: the filter bank and the central frequencies of each filter
    """
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    # ------------------------
    # Compute the filter bank
    # ------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    frequences = numpy.zeros(nfilt + 2, dtype=PARAM_TYPE)
    if nlogfilt == 0:
        linsc = (maxfreq - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt + 2] = lowfreq + numpy.arange(nlinfilt + 2) * linsc
    elif nlinfilt == 0:
        low_mel = hz2mel(lowfreq)
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2)
        # mels[nlinfilt:]
        melsc = (max_mel - low_mel) / (nfilt + 1)
        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences = mel2hz(mels)
    else:
        # Compute linear filters on [0;1000Hz]
        linsc = (min([midfreq, maxfreq]) - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
        # Compute log-linear filters on [1000;maxfreq]
        low_mel = hz2mel(min([1000, maxfreq]))
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2, dtype=PARAM_TYPE)
        melsc = (max_mel - low_mel) / (nlogfilt + 1)

        # Verify that mel2hz(melsc)>linsc
        while mel2hz(melsc) < linsc:
            # in this case, we add a linear filter
            nlinfilt += 1
            nlogfilt -= 1
            frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
            low_mel = hz2mel(frequences[nlinfilt - 1] + 2 * linsc)
            max_mel = hz2mel(maxfreq)
            mels = numpy.zeros(nlogfilt + 2, dtype=PARAM_TYPE)
            melsc = (max_mel - low_mel) / (nlogfilt + 1)

        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences[nlinfilt:] = mel2hz(mels)

    heights = 2. / (frequences[2:] - frequences[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, int(numpy.floor(nfft / 2)) + 1), dtype=PARAM_TYPE)
    # FFT bins (in Hz)
    n_frequences = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = frequences[i]
        cen = frequences[i + 1]
        hi = frequences[i + 2]

        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1, numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        left_slope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1,
                           min(numpy.floor(hi * nfft / fs) + 1, nfft), dtype=numpy.int)
        right_slope = heights[i] / (hi - cen)
        fbank[i][lid] = left_slope * (n_frequences[lid] - low)
        fbank[i][rid[:-1]] = right_slope * (hi - n_frequences[rid[:-1]])

    return fbank, frequences


def power_spectrum(input_sig,
                   fs=8000,
                   win_time=0.025,
                   shift=0.01,
                   prefac=0.97):
    """
    Compute the power spectrum of the signal.
    :param input_sig:
    :param fs:
    :param win_time:
    :param shift:
    :param prefac:
    :return:
    """
    window_length = int(round(win_time * fs))
    overlap = window_length - int(shift * fs)
    framed = framing(input_sig, window_length, win_shift=window_length-overlap).copy()
    # Pre-emphasis filtering is applied after framing to be consistent with stream processing
    framed = pre_emphasis(framed, prefac)
    l = framed.shape[0]
    n_fft = 2 ** int(numpy.ceil(numpy.log2(window_length)))
    # Windowing has been changed to hanning which is supposed to have less noisy sidelobes
    # ham = numpy.hamming(window_length)
    window = numpy.hanning(window_length)

    spec = numpy.ones((l, int(n_fft / 2) + 1), dtype=PARAM_TYPE)
    log_energy = numpy.log((framed**2).sum(axis=1))
    dec = 500000
    start = 0
    stop = min(dec, l)
    while start < l:
        ahan = framed[start:stop, :] * window
        mag = numpy.fft.rfft(ahan, n_fft, axis=-1)
        spec[start:stop, :] = mag.real**2 + mag.imag**2
        start = stop
        stop = min(stop + dec, l)

    return spec, log_energy


def framing(sig, win_size, win_shift=1, context=(0, 0), pad='zeros'):
    """
    :param sig: input signal, can be mono or multi dimensional
    :param win_size: size of the window in term of samples
    :param win_shift: shift of the sliding window in terme of samples
    :param context: tuple of left and right context
    :param pad: can be zeros or edge
    """
    dsize = sig.dtype.itemsize
    if sig.ndim == 1:
        sig = sig[:, numpy.newaxis]
    # Manage padding
    c = (context, ) + (sig.ndim - 1) * ((0, 0), )
    _win_size = win_size + sum(context)
    shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
    strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    if pad == 'zeros':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                                  shape=shape,
                                                  strides=strides).squeeze()
    elif pad == 'edge':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'edge'),
                                                  shape=shape,
                                                  strides=strides).squeeze()


def pre_emphasis(input_sig, pre):
    """Pre-emphasis of an audio signal.
    :param input_sig: the input vector of signal to pre emphasize
    :param pre: value that defines the pre-emphasis filter. 
    """
    if input_sig.ndim == 1:
        return (input_sig - numpy.c_[input_sig[numpy.newaxis, :][..., :1],
                                     input_sig[numpy.newaxis, :][..., :-1]].squeeze() * pre)
    else:
        return input_sig - numpy.c_[input_sig[..., :1], input_sig[..., :-1]] * pre


def mfcc(input_sig,
         lowfreq=100, maxfreq=8000,
         nlinfilt=0, nlogfilt=24,
         nwin=0.025,
         fs=16000,
         nceps=13,
         shift=0.01,
         get_spec=False,
         get_mspec=False,
         prefac=0.97):
    """Compute Mel Frequency Cepstral Coefficients.

    :param input_sig: input signal from which the coefficients are computed.
            Input audio is supposed to be RAW PCM 16bits
    :param lowfreq: lower limit of the frequency band filtered. 
            Default is 100Hz.
    :param maxfreq: higher limit of the frequency band filtered.
            Default is 8000Hz.
    :param nlinfilt: number of linear filters to use in low frequencies.
            Default is 0.
    :param nlogfilt: number of log-linear filters to use in high frequencies.
            Default is 24.
    :param nwin: length of the sliding window in seconds
            Default is 0.025.
    :param fs: sampling frequency of the original signal. Default is 16000Hz.
    :param nceps: number of cepstral coefficients to extract. 
            Default is 13.
    :param shift: shift between two analyses. Default is 0.01 (10ms).
    :param get_spec: boolean, if true returns the spectrogram
    :param get_mspec:  boolean, if true returns the output of the filter banks
    :param prefac: pre-emphasis filter value

    :return: the cepstral coefficients in a ndaray as well as 
            the Log-spectrum in the mel-domain in a ndarray.

    .. note:: MFCC are computed as follows:
        
            - Pre-processing in time-domain (pre-emphasizing)
            - Compute the spectrum amplitude by windowing with a Hamming window
            - Filter the signal in the spectral domain with a triangular filter-bank, whose filters are approximatively
               linearly spaced on the mel scale, and have equal bandwith in the mel scale
            - Compute the DCT of the log-spectrom
            - Log-energy is returned as first coefficient of the feature vector.
    
    For more details, refer to [Davis80]_.
    """
    # Compute power spectrum
    spec, log_energy = power_spectrum(input_sig,
                                      fs,
                                      win_time=nwin,
                                      shift=shift,
                                      prefac=prefac)
    # Filter the spectrum through the triangle filter-bank
    n_fft = 2 ** int(numpy.ceil(numpy.log2(int(round(nwin * fs)))))
    fbank = trfbank(fs, n_fft, lowfreq, maxfreq, nlinfilt, nlogfilt)[0]

    mspec = numpy.log(numpy.dot(spec, fbank.T))   # A tester avec log10 et log
    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    # The C0 term is removed as it is the constant term
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, 1:nceps + 1]
    lst = list()
    lst.append(ceps)
    lst.append(log_energy)
    if get_spec:
        lst.append(spec)
    else:
        lst.append(None)
        del spec
    if get_mspec:
        lst.append(mspec)
    else:
        lst.append(None)
        del mspec

    return lst
