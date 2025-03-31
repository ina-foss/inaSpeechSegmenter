"""
VBx Features Module
-------------------
This module implements a collection of functions for audio feature extraction and processing, 
with functions essential  for preparing audio data for speech processing tasks 
such as speaker diarization and recognition, as well as other audio analysis applications.

It also encapsulates the core feature extraction procedures required for transforming raw audio
signals into features suitable for speech analysis and speaker recognition pipelines.
"""

#!/usr/bin/env python

# Copyright Brno University of Technology (burget@fit.vutbr.cz)
# Licensed under the Apache License, Version 2.0 (the "License")
# From VBHMM x-vectors Diarization (aka VBx)
# Available on : https://github.com/BUTSpeechFIT/VBx/blob/master/VBx/features.py

import numpy as np


def framing(a, window, shift=1):
    """Creates overlapping frames from an input array using numpy's stride tricks without copying data.
    Each frame has a specified window length and shift, useful for efficient signal processing tasks."""
    
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# Mel and inverse Mel scale warping functions

def mel(x):
    # mel(x): Converts a linear frequency (in Hz) to the Mel scale, 
    # which models human auditory perception by emphasizing frequencies that the ear is more sensitive to.
    return 1127. * np.log(1. + x/700.)

def mel_inv(x):
    # mel_inv(x): Converts a Mel-scaled frequency back to a linear frequency (in Hz) 
    # using the inverse logarithmic transformation.
    return (np.exp(x/1127.) - 1.) * 700.


def preemphasis(x, coef=0.97):
    """
    preemphasis(x, coef=0.97):
    Applies a pre-emphasis filter (high-pass filter to amplify high-frequency components of a signal).
    This is used to improve the signal-to-noise ratio.
    Enhancing these components is crucial for effective speech feature extraction, such as in MFCC computation.
    """
    return x - np.c_[x[..., :1], x[..., :-1]] * coef


def mel_fbank_mx(winlen_nfft, fs, NUMCHANS=20, LOFREQ=0.0, HIFREQ=None, warp_fn=mel, inv_warp_fn=mel_inv, htk_bug=True):
    """
    mel_fbank_mx: Generates a Mel filterbank matrix as a 2D array with dimensions (NFFT/2+1 x NUMCHANS).
    
    A Mel filterbank is a collection of overlapping triangular filters that are applied to the power spectrum of a signal,
    transforming it into a representation that aligns with human auditory perception. The filters emphasize frequency
    bands that humans are more sensitive to (the Mel scale), making them a key component in speech processing tasks
    such as MFCC (Mel Frequency Cepstral Coefficients) extraction.
    
    Parameters:
    - winlen_nfft: Determines the FFT length (NFFT) for spectral analysis. If positive, it is rounded up to the next higher
                   power of two (HTK-compatible). If negative, NFFT is set to the absolute value.
    - fs: Sampling frequency in Hz.
    - NUMCHANS: Number of filters (bands) in the filterbank.
    - LOFREQ: Lower bound frequency (Hz) where the first filter starts.
    - HIFREQ: Upper bound frequency (Hz) where the last filter ends (default is fs/2 if not specified).
    - warp_fn: Function for converting linear frequencies to the Mel scale.
    - inv_warp_fn: Inverse function to convert Mel frequencies back to linear scale.
    """
    HIFREQ = 0.5 * fs if not HIFREQ else HIFREQ
    nfft = 2**int(np.ceil(np.log2(winlen_nfft))) if winlen_nfft > 0 else -int(winlen_nfft)

    fbin_mel = warp_fn(np.arange(nfft / 2 + 1, dtype=float) * fs / nfft)
    cbin_mel = np.linspace(warp_fn(LOFREQ), warp_fn(HIFREQ), NUMCHANS + 2)
    cind = np.floor(inv_warp_fn(cbin_mel) / fs * nfft).astype(int) + 1
    mfb = np.zeros((len(fbin_mel), NUMCHANS))
    for i in range(NUMCHANS):
        mfb[cind[i]:cind[i+1], i] = (cbin_mel[i] - fbin_mel[cind[i]:cind[i+1]]) / (cbin_mel[i] - cbin_mel[i+1])
        mfb[cind[i+1]:cind[i+2], i] = (cbin_mel[i+2] - fbin_mel[cind[i+1]:cind[i+2]]) / \
                                      (cbin_mel[i+2] - cbin_mel[i+1])
    if LOFREQ > 0.0 and float(LOFREQ) / fs * nfft + 0.5 > cind[0] and htk_bug:
        mfb[cind[0], :] = 0.0  # Just to be HTK compatible
    return mfb


def fbank_htk(x, window, noverlap, fbank_mx, nfft=None, _E=None,
              USEPOWER=False, RAWENERGY=True, PREEMCOEF=0.97, ZMEANSOURCE=False,
              ENORMALISE=True, ESCALE=0.1, SILFLOOR=50.0, USEHAMMING=True):
    """
    fbank_htk: Computes log Mel-filter bank features from an input signal.
    
    This function processes an audio signal by framing it into overlapping windows, applying a window function,
    performing an FFT, and then mapping the resulting power spectrum through a Mel filterbank (as provided by fbank_mx).
    The output is a NUMCHANS-by-M matrix, where NUMCHANS is the number of filter bands and M is the number of frames
    extracted from the signal. Optionally, the function can include the energy of each frame as an extra coefficient
    (either at the beginning or the end of the feature vector), similar to the energy handling in HTK.
    
    Parameters:
    - x: The input audio signal.
    - window: The length (in samples) of each frame or a vector of window weights (overriding the default windowing function).
    - noverlap: The number of overlapping samples between consecutive frames.
    - fbank_mx: The Mel filterbank matrix (output of mel_fbank_mx), which must be compatible with the chosen FFT size.
    - nfft: The number of FFT samples; if not specified, it defaults to the next power of two of the window length.
    - _E: Option to include frame energy in the output ("first", "last", or None).
    
    Additional options follow HTK conventions, such as energy normalization and dithering, to facilitate robust feature extraction.
    """
    if type(USEPOWER) == bool:
        USEPOWER += 1
    if np.isscalar(window):
        window = np.hamming(window) if USEHAMMING else np.ones(window)
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(window.size)))
    x = framing(x.astype("float"), window.size, window.size-noverlap).copy()
    if ZMEANSOURCE:
        x -= x.mean(axis=1)[:, np.newaxis]
    if _E is not None and RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    if PREEMCOEF is not None:
        x = preemphasis(x, PREEMCOEF)
    x *= window
    if _E is not None and not RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    x = np.fft.rfft(x, nfft)
    x = x.real**2 + x.imag**2
    if USEPOWER != 2:
        x **= 0.5 * USEPOWER
    x = np.log(np.maximum(1.0, np.dot(x, fbank_mx)))
    if _E is not None and ENORMALISE:
        energy = (energy - energy.max()) * ESCALE + 1.0
        min_val = -np.log(10**(SILFLOOR/10.)) * ESCALE + 1.0
        energy[energy < min_val] = min_val

    return np.hstack(([energy[:, np.newaxis]] if _E == "first" else []) + [x] +
                     ([energy[:, np.newaxis]] if (_E in ["last", True]) else []))


def povey_window(winlen):
    return np.power(0.5 - 0.5*np.cos(np.linspace(0, 2*np.pi, winlen)), 0.85)


def add_dither(x, level=8):
    return x + level * (np.random.rand(*x.shape)*2 - 1)


def cmvn_floating_kaldi(x, LC, RC, norm_vars=True):
    """Mean and variance normalization over a floating window.
    x is the feature matrix (nframes x dim)
    LC, RC are the number of frames to the left and right defining the floating
    window around the current frame. This function uses Kaldi-like treatment of
    the initial and final frames: Floating windows stay of the same size and
    for the initial and final frames are not centered around the current frame
    but shifted to fit in at the beginning or the end of the feature segment.
    Global normalization is used if nframes is less than LC+RC+1.
    """
    N, dim = x.shape
    win_len = min(len(x), LC+RC+1)
    win_start = np.maximum(np.minimum(np.arange(-LC, N-LC), N-win_len), 0)
    f = np.r_[np.zeros((1, dim)), np.cumsum(x, 0)]
    x = x - (f[win_start+win_len] - f[win_start]) / win_len
    if norm_vars:
        f = np.r_[np.zeros((1, dim)), np.cumsum(x**2, 0)]
        x /= np.sqrt((f[win_start+win_len] - f[win_start]) / win_len)
    return x
