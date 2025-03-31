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

"""
PraatTextGrid File Comment
---------------------------
This module utilizes the PraatTextGrid class from the pytextgrid package to
handle the creation and manipulation of Praat TextGrid files.

A PraatTextGrid is a representation of the TextGrid file format used by the
Praat software, which is widely employed in phonetics and speech analysis.
TextGrid files are used to annotate audio recordings by marking time intervals
(e.g., words, phonemes, or other speech segments) on one or more tiers. Each tier
can contain multiple intervals, with each interval defined by a start time, stop time,
and a corresponding label.
"""

import pandas as pd
from pytextgrid.PraatTextGrid import PraatTextGrid, Interval, Tier


def seg2csv(lseg, fout=None):
    """
    Function: seg2csv
    -----------------
    Converts a list of segmentation data into a CSV file.
    
    Parameters:
    - lseg: A list of segments, where each segment is represented as a tuple or list
            containing three elements: (label, start_time, stop_time).
    - fout: (Optional) The file path or file-like object where the CSV output should be saved.
            If not provided, the function may return the DataFrame for further processing.
    
    This function is useful for converting annotated segmentation data into a CSV format
    for easier analysis, sharing, or further processing in data analysis pipelines.
    """
    df = pd.DataFrame.from_records(lseg, columns=['labels', 'start', 'stop'])
    df.to_csv(fout, sep='\t', index=False)

def seg2textgrid(lseg, fout=None):
    """
    Function: seg2textgrid
    ----------------------
    Converts a list of segmentation data into a Praat TextGrid file.
    
    Parameters:
    - lseg: A list of segments, where each segment is a tuple or list containing three elements:
            (label, start_time, stop_time). The list should be ordered by time.
    - fout: (Optional) The file path or file-like object where the resulting TextGrid file will be saved.
            If not provided, the function may use a default behavior (e.g., returning the TextGrid object).
    
    This function is essential for converting speech segmentation data into a format compatible with Praat,
    allowing for further audio annotation analysis.
    """
    tier = Tier(name='inaSpeechSegmenter')
    for label, start, stop in lseg:
        tier.append(Interval(start, stop, label))
    ptg = PraatTextGrid(xmin=lseg[0][1], xmax=lseg[-1][2])
    ptg.append(tier)
    ptg.save(fout)
