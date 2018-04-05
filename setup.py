1#!/usr/bin/env python
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
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

KEYWORDS = 'speech-segmentation audio-analysis speaker-gender-segmentation music-detection speech-music gender-equality gender-classification'

CLASSIFIERS=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Topic :: Multimedia :: Sound/Audio',
    'Topic :: Multimedia :: Sound/Audio :: Analysis',
    'Topic :: Multimedia :: Sound/Audio :: Speech',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Sociology',
]

DESCRIPTION='CNN-based audio segmentation toolkit. Allows to detect speech, music and speaker gender. Has been designed for large scale gender equality studies based on speech time per gender.'

setup(
    name = "inaSpeechSegmenter",
    version = "0.0.1",
    author = "David Doukhan",
    author_email = "david.doukhan@gmail.com",
    description = DESCRIPTION,
    license = "MIT",
    install_requires=['numpy', 'keras', 'scikit-image', 'sidekit', 'pyannote.algorithms'],
 #   keywords = "example documentation tutorial",
    url = "https://github.com/ina-foss/inaSpeechSegmenter",
#    packages=['inaSpeechSegmenter'],
    keywords = KEYWORDS.split(' '),
    packages = find_packages(),
    package_data = {'inaSpeechSegmenter': ['*.hdf5']},
    include_package_data = True,
    
    long_description=read('README.md'),
    scripts=[os.path.join('scripts', script) for script in \
             ['ina_speech_segmenter.py']],
    classifiers=CLASSIFIERS,
)
