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
from setuptools import setup, find_packages


KEYWORDS = '''
speech-segmentation
audio-analysis
music-detection
speech-music
gender-equality
gender-classification
speaker-gender
speech
music
voice-activity-detection'''.strip().split('\n')

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

DESCRIPTION='CNN-based audio segmentation toolkit. Does voice activity detection, music recognition, speaker gender recognition.'
LONGDESCRIPTION='''Split audio signal into homogeneous zones of speech and music, and detect speaker gender.
For further details, see the following publication:
Doukhan, D., Carrive, J., Vallet, F., Larcher, A., Meignier, S. (2018).
An Open-Source Speaker Gender Detection Framework for Monitoring Gender Equality.
in 2018 IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP)
'''

setup(
    name = "inaSpeechSegmenter",
    version = "0.0.3",
    author = "David Doukhan",
    author_email = "david.doukhan@gmail.com",
    description = DESCRIPTION,
    license = "MIT",
    install_requires=['numpy', 'keras', 'scikit-image', 'sidekit', 'pyannote.algorithms'],
 #   keywords = "example documentation tutorial",
    url = "https://github.com/ina-foss/inaSpeechSegmenter",
#    packages=['inaSpeechSegmenter'],
    keywords = KEYWORDS,
    packages = find_packages(),
    package_data = {'inaSpeechSegmenter': ['*.hdf5']},
    include_package_data = True,
    data_files = ['LICENSE',
                  'inaSpeechSegmenter/keras_speech_music_cnn.hdf5',
                  'inaSpeechSegmenter/keras_male_female_cnn.hdf5',
                  'API_Tutorial.ipynb'],
    long_description=LONGDESCRIPTION,
    scripts=[os.path.join('scripts', script) for script in \
             ['ina_speech_segmenter.py']],
    classifiers=CLASSIFIERS,
    python_requires='>=3',
)
