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

import sys
import Pyro4
import numpy as np
import pandas as pd


@Pyro4.expose
class GenderJobServer(object):
    def __init__(self, df):
        self.lsource = list(df.source_path)
        self.ldest = list(df.dest_path)
        self.i = 0
        
    def get_job(self, msg):
        print('job %d: %s' % (self.i, msg))
        self.i += 1
        return (self.lsource.pop(0), self.ldest.pop(0))

    def get_njobs(self, msg, nbjobs=20):
        print('jobs %d-%d: %s' % (self.i, self.i + nbjobs, msg))
        ret = (self.lsource[:nbjobs], self.ldest[:nbjobs])
        if len(ret[0]) == 0:
            print('All jobs dispatched')
        self.lsource = self.lsource[nbjobs:]
        self.ldest = self.ldest[nbjobs:]
        self.i += nbjobs
        return ret



if __name__ == '__main__':
    # full name of the host to be used by remote clients
    Pyro4.config.HOST = sys.argv[1]
    # csv configuration file with 2 columns: source_path, dest_path
    df = pd.read_csv(sys.argv[2]).sample(frac=1).reset_index(drop=True)

    print('random source & dest path:', df.source_path[0], ' ',df.dest_path[0])
    print('number of files to process:', len(df))
    
    
    daemon = Pyro4.Daemon()                # make a Pyro daemon\n",
    uri = daemon.register(GenderJobServer(df))   # register the greeting maker as a Pyro object\n",
    print("Ready. Object uri =", uri) 
    daemon.requestLoop()
