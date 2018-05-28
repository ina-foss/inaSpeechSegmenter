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

import argparse
import glob
import os
import warnings

# TODO:
# * Allow the selection of a custom ffmpeg binary
# * allow to use an external activity or speech music segmentations
# * describe URL management in help and interference with glob

# Configure command line parsing
parser = argparse.ArgumentParser(description='Do Speech/Music and Male/Female segmentation. Store segmentations into CSV files')
parser.add_argument('-i', '--input', nargs='+', help='Input media to analyse. May be a full path to a media (/home/david/test.mp3), a list of full paths (/home/david/test.mp3 /tmp/mymedia.avi), a regex input pattern ("/home/david/myaudiobooks/*.mp3"), an url with http* protocol (http://url_of_the_file), or a .txt file containing any of the those.', required=True)
parser.add_argument('-o', '--output_directory', help='Directory used to store segmentations. Resulting segmentations have same base name as the corresponding input media, with csv extension. Ex: mymedia.MPG will result in mymedia.csv', required=True)
parser.add_argument('-b', '--ffmpeg_binary', help='Your custom binary of ffmpeg', required=False)
args = parser.parse_args()

# Preprocess arguments and check their consistency
input_files = []
for e in args.input:
    if e.startswith("http"):
        input_files += [e]
    else:
        input_files += glob.glob(e)
assert len(input_files) > 0, "No existing media selected for analysis! Bad values provided to -i ({})".format(args.input)

odir = args.output_directory.strip(" \t\n\r").rstrip('/')
assert os.access(odir, os.W_OK), 'Directory %s is not writable!' % odir

# Do processings
from inaSpeechSegmenter import Segmenter, seg2csv, to_parse

# load neural network into memory, may last few seconds
seg = Segmenter()

# case of a file of files
files = to_parse(input_files)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    #print('processing file %d/%d: %s' % (i+1, len(input_files), e))
    base = [os.path.splitext(os.path.basename(e)) for e in files]
    base = [base[i][0] for i in range(len(base))]
    if len(odir) > 0:
        fout = ['%s/%s.csv' % (odir, elem) for elem in base]
    seg2csv(seg(files), fout)





