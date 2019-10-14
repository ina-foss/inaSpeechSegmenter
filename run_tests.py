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

class TestInaSpeechSegmenter(unittest.TestCase):
    def test_tf(self):
        # tensorflow should be installed and is not included
        # in inaspeechsegmenter dependencies
        import tensorflow as tf
    
    def test_import(self):
        from inaSpeechSegmenter import Segmenter
        seg = Segmenter()

    def test_execution(self):
        # if this test fails, then you should check to correctness of your
        # tensorflow installation
        from inaSpeechSegmenter import Segmenter
        seg = Segmenter()
        ret = seg('./media/musanmix.mp3')
        
    def test_short(self):
        from inaSpeechSegmenter import Segmenter
        seg = Segmenter()
        seg('./media/0021.mp3')

    def test_boundaries(self):

        def seg2str(iseg, tseg):
            label, start, stop  = tseg
            return 'seg %d <%s, %f, %f>' % (iseg, label, start, stop)
        
        from inaSpeechSegmenter import Segmenter
        seg = Segmenter()
        ret = seg('./media/musanmix.mp3')
        for i in range(len(ret) -1):
            curstop = ret[i][2]
            nextstart = ret[i+1][1]
            self.assertEqual(curstop, nextstart,
                             '%s VS %s' % (seg2str(i, ret[i]), seg2str(i+1, ret[i+1])))

    def test_processingresult(self):
        from inaSpeechSegmenter import Segmenter
        seg = Segmenter()
        ret = seg('./media/musanmix.mp3')
        ref = [('Music', 0.0, 22.48), ('NOACTIVITY', 22.48, 29.080000000000002), ('Male', 29.080000000000002, 32.480000000000004), ('Music', 32.480000000000004, 52.800000000000004), ('NOACTIVITY', 52.800000000000004, 54.78), ('Music', 54.78, 55.74), ('NOACTIVITY', 55.74, 63.34), ('Male', 63.34, 68.26), ('NOACTIVITY', 68.26, 68.92), ('Male', 68.92, 71.60000000000001), ('NOACTIVITY', 71.60000000000001, 72.0), ('Male', 72.0, 73.82000000000001), ('NOACTIVITY', 73.82000000000001, 74.5)]
        self.assertEqual(ref, ret)
            
if __name__ == '__main__':
    msg = """
    Testing InaSpeechSegmenter
    Currently: only 2 tests are supposed to work correctly: test_import and test_execution.
    The remaining tests show errors that should be fixed soon
"""
    print(msg)
    
    unittest.main()
