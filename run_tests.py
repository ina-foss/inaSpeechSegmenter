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


import unittest
import os
from inaSpeechSegmenter import Segmenter

class TestInaSpeechSegmenter(unittest.TestCase):
    
    def test_init(self):
        seg = Segmenter()

    def test_execution(self):
        # if this test fails, then you should check to correctness of your
        # tensorflow installation
        seg = Segmenter()
        ret = seg('./media/musanmix.mp3')
        
    def test_short(self):
        seg = Segmenter(vad_engine='sm')
        ret = seg('./media/0021.mp3')
        ref = [('male', 0, 0.66)]
        self.assertEqual(ref, ret)

    def test_boundaries(self):

        def seg2str(iseg, tseg):
            label, start, stop  = tseg
            return 'seg %d <%s, %f, %f>' % (iseg, label, start, stop)
        
        seg = Segmenter()
        ret = seg('./media/musanmix.mp3')
        for i in range(len(ret) -1):
            curstop = ret[i][2]
            nextstart = ret[i+1][1]
            self.assertEqual(curstop, nextstart,
                             '%s VS %s' % (seg2str(i, ret[i]), seg2str(i+1, ret[i+1])))

    def test_processingresult(self):
        seg = Segmenter(vad_engine='sm')
        ret = seg('./media/musanmix.mp3')
        ref = [('music', 0.0, 22.48), ('noEnergy', 22.48, 29.080000000000002), ('male', 29.080000000000002, 32.480000000000004), ('music', 32.480000000000004, 52.800000000000004), ('noEnergy', 52.800000000000004, 54.78), ('music', 54.78, 55.74), ('noEnergy', 55.74, 63.34), ('male', 63.34, 68.26), ('noEnergy', 68.26, 68.92), ('male', 68.92, 71.60000000000001), ('noEnergy', 71.60000000000001, 72.0), ('male', 72.0, 73.82000000000001), ('noEnergy', 73.82000000000001, 74.5)]
        self.assertEqual(ref, ret)

    def test_program(self):
        ret = os.system('CUDA_VISIBLE_DEVICES="" ./scripts/ina_speech_segmenter.py -i ./media/0021.mp3 -o ./')
        self.assertEqual(ret, 0, 'ina_speech_segmenter returned error code %d' % ret)

    def test_startsec(self):
        # test start_sec argument
        seg = Segmenter()
        start_sec = 2.
        for lab, start, stop in seg('./media/musanmix.mp3', start_sec=start_sec):
            self.assertGreaterEqual(start, start_sec)
            self.assertGreaterEqual(stop, start_sec)

    def test_stopsec(self):
        # test stop_sec argument
        seg = Segmenter()
        stop_sec = 5.
        for lab, start, stop in seg('./media/musanmix.mp3', stop_sec=stop_sec):
            self.assertLessEqual(stop, stop_sec)
            self.assertLessEqual(start, stop_sec)
        
if __name__ == '__main__':
    unittest.main()
