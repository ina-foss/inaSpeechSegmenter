#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2018 Ina (David Doukhan, Eliott Lechapt - http://www.ina.fr/)

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

import argparse
import glob
import os
import warnings

# TODO
# * allow to use external activity or speech music segmentations
# * describe URL management in help and interference with glob

description = """Do Speech/Music(/Noise) and Male/Female segmentation and store segmentations into CSV files. Segments labelled 'noEnergy' are discarded from music, noise, speech and gender analysis. 'speech', 'male' and 'female' labels include speech over music and speech over noise. 'music' and 'noise' labels are pure segments that are not supposed to contain speech.
"""
epilog="""
Detailled description of this framework are presented in the following studies:
* Doukhan, D., Lechapt, E., Evrard, M., & Carrive, J. (2018). Ina’s mirex 2018 music and speech detection system. Music Information Retrieval Evaluation eXchange (MIREX 2018).
* Doukhan, D., Carrive, J., Vallet, F., Larcher, A., & Meignier, S. (2018, April). An open-source speaker gender detection framework for monitoring gender equality. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5214-5218). IEEE.
* Doukhan, D., Devauchelle, S., Girard-Monneron, L., Chávez Ruz, M., Chaddouk, V., Wagner, I., Rilliard, A. (2023) Voice Passing : a Non-Binary Voice Gender Prediction System for evaluating Transgender voice transition. Proc. INTERSPEECH 2023, 5207-5211, doi: 10.21437/Interspeech.2023-1835
"""

vad_engine_help="""Voice activity detection (VAD) engine to be used (default: 'smn').
'smn' split signal into 'speech', 'music' and 'noise' (better).
'sm' split signal into 'speech' and 'music'.
"""

gender_engine_help = """Speech Gender Segmentation engine to be used after VAD engine (default: 'ic18').
if set to 'none', no speech gender segmentation engine will be used, and corresponding labels will be labelled as 'speech' (faster).
Otherwise, speech segments will be splitted into 'male' and 'female' segments using 
ICASSP 2018 CNN model (ic18) or Interspeech 2023 Xvector model (is23)
"""

# Configure command line parsing
parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-i', '--input', nargs='+', help='Input media to analyse. May be a full path to a media (/home/david/test.mp3), a list of full paths (/home/david/test.mp3 /tmp/mymedia.avi), a regex input pattern ("/home/david/myaudiobooks/*.mp3"), an url with http protocol (http://url_of_the_file)', required=True)
parser.add_argument('-o', '--output_directory', help='Directory used to store segmentations. Resulting segmentations have same base name as the corresponding input media, with csv extension. Ex: mymedia.MPG will result in mymedia.csv', required=True)
parser.add_argument('-s', '--batch_size', type=int, default=32, help="(default: 32 - we recommend 1024). Size of batches to be sent to the GPU. Larger values allow faster processings, but require GPU with more memories. Default 32 size is fine even with a baseline laptop GPU.")
parser.add_argument('-d', '--vad_engine', choices=['sm', 'smn'], default='smn', help = vad_engine_help)
parser.add_argument('-g', '--gender_engine', choices = ['none', 'ic18', 'is23'], default = 'ic18', help = gender_engine_help)
parser.add_argument('-b', '--ffmpeg_binary', default='ffmpeg', help='Your custom binary of ffmpeg', required=False)
parser.add_argument('-e', '--export_format', choices = ['csv', 'textgrid'], default='csv', help="(default: 'csv'). If set to 'csv', result will be exported in csv. If set to 'textgrid', results will be exported to praat Textgrid")
parser.add_argument('-r', '--energy_ratio', default=0.03, type=float, help="(default: 0.03). Energetic threshold used to detect activity (percentage of mean energy of the signal)")

args = parser.parse_args()

# Preprocess arguments and check their consistency
input_files = []
for e in args.input:
    if e.startswith("http"):
        input_files += [e]
    else:
        input_files += glob.glob(e)
assert len(input_files) > 0, 'No existing media selected for analysis! Bad values provided to -i (%s)' % args.input

odir = args.output_directory.strip(" \t\n\r").rstrip('/')
assert os.access(odir, os.W_OK), 'Directory %s is not writable!' % odir

# Do processings
from inaSpeechSegmenter import Segmenter

# load neural network into memory, may last few seconds
seg = Segmenter(vad_engine=args.vad_engine, gender_engine=args.gender_engine, ffmpeg=args.ffmpeg_binary, energy_ratio=args.energy_ratio, batch_size=args.batch_size)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    base = [os.path.splitext(os.path.basename(e))[0] for e in input_files]
    output_files = [os.path.join(odir, e + '.' + args.export_format) for e in base]
    seg.batch_process(input_files, output_files, verbose=True, output_format=args.export_format)

