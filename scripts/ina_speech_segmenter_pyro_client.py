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


import Pyro4
import sys
import os
import socket
import time

from inaSpeechSegmenter import Segmenter, seg2csv

def myprocess(in_fname, out_fname):
    
    if os.path.isfile(out_fname):
        return 'already done'

    dname = os.path.dirname(out_fname)
    if not os.path.isdir(dname):
        os.makedirs(dname)

    results = g(in_fname)
    seg2csv(results, out_fname)
    return 0

if __name__ == '__main__':
    dname = os.path.dirname(os.path.realpath(__file__))

    hostname = socket.gethostname()

    uri = sys.argv[1]
    jobserver = Pyro4.Proxy(uri)

    b = time.time()
    ret = -1
    outname = 'init'
    
    g = Segmenter()
    
    while True:
        url, outname = jobserver.get_job('%s %f %s %s' % (hostname, time.time() - b, ret, outname))
            
        b = time.time()

        print(url, outname)
        
        try:
            ret =  myprocess(url, outname)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            ret = 'error'
