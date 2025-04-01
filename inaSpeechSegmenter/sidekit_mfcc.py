# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT, a Python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# The code in this file has been adapted from SIDEKIT source files (frontend/features.py, frontend/io.py, frontend/vad.py)
# to provide core frontend functionality for processing audio signals. It includes methods for Mel-scale conversion,
# filterbank construction, power spectrum analysis, framing, pre-emphasis filtering, and MFCC extraction.
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
#import soundfile
#import scipy
from scipy.fftpack.realtransforms import dct


__author__ = "Anthony Larcher and Sylvain Meignier"
__copyright__ = "Copyright 2014-2021 Anthony Larcher and Sylvain Meignier"
__license__ = "LGPL"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'




PARAM_TYPE = numpy.float32

def hz2mel(f, htk=True):
    """
    Convert frequencies from Hertz (Hz) to the Mel scale.

    Parameters:
        f (array-like): a frequency or an array of frequencies in Hz.
        htk (bool): if True, use the HTK formula; otherwise, use the formula that matches Slaney's Auditory Toolbox.

    Returns:
        numpy.ndarray or float: The corresponding value(s) on the Mel scale.
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
    """
    Compute a triangular filterbank for cepstral coefficient extraction.

    This function constructs a filterbank where each filter is defined by a triangle in the spectral domain.
    The filters are spaced linearly in the low-frequency range and logarithmically in the high-frequency range,
    based on the specified numbers of linear (nlinfilt) and log-linear (nlogfilt) filters. The filterbank is 
    computed to map the FFT bins (derived from nfft points) into the Mel scale for robust feature extraction.
    
    Parameters:
        fs (int): sampling frequency of the original signal.
        nfft (int): number of points for the Fourier Transform.
        lowfreq (float): lower frequency bound for the filterbank.
        maxfreq (float): upper frequency bound for the filterbank.
        nlinfilt (int): number of filters with linear spacing in the low frequencies.
        nlogfilt (int): number of filters with logarithmic spacing in the high frequencies.
        midfreq (float): frequency boundary between linear and log-linear filter spacing (default is 1000 Hz).

    Returns:
        fbank (numpy.ndarray): The computed filterbank matrix, where each row corresponds to a triangular filter
                               applied in the FFT domain.
        frequences (numpy.ndarray): The central frequencies of the triangular filters.
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

        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1, numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int32)
        left_slope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1,
                           min(numpy.floor(hi * nfft / fs) + 1, nfft), dtype=numpy.int32)
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
    Computes the power spectrum and log energy of an audio signal.

    This function first frames the input signal using a Hanning window with specified window duration 
    (win_time) and shift (in seconds). Pre-emphasis filtering is applied after framing for consistent stream processing.
    The function then computes the FFT on each frame to obtain the power spectrum and calculates the log energy
    for each frame.

    Parameters:
        input_sig (numpy.ndarray): the input audio signal.
        fs (int): sampling frequency in Hz. Default is 8000 Hz.
        win_time (float): duration of each frame in seconds (e.g., 0.025 for 25 ms).
        shift (float): time shift between successive frames in seconds (e.g., 0.01 for 10 ms).
        prefac (float): pre-emphasis coefficient applied to each frame.

    Returns:
        spec (numpy.ndarray): 2D array of power spectrum values for each frame.
        log_energy (numpy.ndarray): 1D array of the log energy for each frame.
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
    Extracts overlapping frames from an input signal using efficient stride tricks.

    Parameters:
        sig (numpy.ndarray): the input signal, which can be mono or multi-dimensional.
        win_size (int): the size of each frame in samples.
        win_shift (int): the number of samples to shift the window for the next frame.
        context (tuple): a tuple (left, right) specifying additional context samples to include with each frame.
        pad (str): padding method to use when extending the signal ('zeros' for zero-padding or 'edge' to repeat edge values).

    Returns:
        numpy.ndarray: An array of overlapping frames extracted from the input signal.
    
    The function pads the signal based on the provided context, computes the appropriate shape and strides,
    and then uses numpy's stride_tricks to create a view into the signal without copying data.
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
        return numpy.lib.stride_tricks.as_strided(numpy.pad(sig, c, 'constant', constant_values=(0,)),
                                                  shape=shape,
                                                  strides=strides).squeeze()
    elif pad == 'edge':
        return numpy.lib.stride_tricks.as_strided(numpy.pad(sig, c, 'edge'),
                                                  shape=shape,
                                                  strides=strides).squeeze()


def pre_emphasis(input_sig, pre):
    """
    Applies a pre-emphasis filter to an audio signal.

    This function boosts high-frequency components by subtracting a scaled version of the previous sample from the current sample.
    The pre-emphasis coefficient 'pre' controls the amount of high-frequency amplification, which can improve feature extraction.
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
    """
    Compute Mel Frequency Cepstral Coefficients (MFCCs) and related features from an audio signal.

    This function extracts MFCCs from a RAW PCM 16-bit audio signal by first computing the power spectrum through
    windowing and pre-emphasis filtering, then filtering the spectrum with a triangular filterbank (based on the Mel scale).
    The logarithm of the filterbank energies (mspec) is computed, and the Discrete Cosine Transform (DCT) is applied to 
    obtain the MFCCs (excluding the C0 term, which represents overall energy). Optionally, the function can also return the 
    full spectrogram and the log Mel-spectrum.

    Parameters:
        input_sig (numpy.ndarray): input audio signal (expected as RAW PCM 16-bit).
        lowfreq (float): lower frequency bound for filtering (default 100 Hz).
        maxfreq (float): upper frequency bound for filtering (default 8000 Hz).
        nlinfilt (int): number of linearly spaced filters for low frequencies (default 0).
        nlogfilt (int): number of logarithmically spaced filters for high frequencies (default 24).
        nwin (float): window length in seconds for framing the signal (default 0.025 sec).
        fs (int): sampling frequency of the input signal (default 16000 Hz).
        nceps (int): number of cepstral coefficients to extract (default 13).
        shift (float): time shift between successive frames in seconds (default 0.01 sec).
        get_spec (bool): if True, include the power spectrogram in the output.
        get_mspec (bool): if True, include the log Mel-spectrum in the output.
        prefac (float): pre-emphasis coefficient applied to the signal (default 0.97).

    Returns:
        list: A list containing:
              - MFCCs (numpy.ndarray): The computed cepstral coefficients (excluding the 0th coefficient).
              - Log energy (numpy.ndarray): Logarithm of the frame energies.
              - Spectrogram (numpy.ndarray or None): The power spectrum if get_spec is True, otherwise None.
              - Log Mel-spectrum (numpy.ndarray or None): The log energies of the filterbank if get_mspec is True, otherwise None.

    The MFCCs provide a compact representation of the spectral envelope of the audio signal, which is crucial for
    tasks such as speaker verification, speech recognition, and other audio analysis applications.
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
