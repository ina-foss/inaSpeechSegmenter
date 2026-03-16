
import unittest
import os
import shutil
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inaSpeechSegmenter.segmenter_keras import Segmenter
from inaSpeechSegmenter.vbx_segmenter_keras import VoiceFemininityScoring

class TestKerasInaSpeechSegmenter(unittest.TestCase):
    
    def test_execution_keras(self):
        print("\nTesting Keras Segmenter on musanmix.mp3...")
        # Force download/check of HDF5 models by instantiating
        seg = Segmenter()
        ret = seg('./media/musanmix.mp3')
        print(f"Result (first 3 segments): {ret[:3]}")
        self.assertTrue(len(ret) > 0)
        
    def test_short_keras(self):
        print("\nTesting Keras Segmenter (short file) on 0021.mp3...")
        seg = Segmenter(vad_engine='sm')
        ret = seg('./media/0021.mp3')
        ref = [('male', 0, 0.66)]
        print(f"Result: {ret}")
        self.assertEqual(ref, ret)

    def test_vfs_keras(self):
        print("\nTesting Keras VoiceFemininityScoring on lamartine.wav...")
        media = './media/lamartine.wav'
        desired_vfp_score = 0.534884
        vfs = VoiceFemininityScoring(gd_model_criteria="vfp")
        score, dur, nb = vfs(media)
        print(f"VFS Score: {score}")
        
        np.testing.assert_almost_equal(
            score,
            desired_vfp_score,
            decimal=4,
            err_msg='Voice Femininity Score mismatch (Keras vs Expected)'
        )

if __name__ == '__main__':
    unittest.main()
