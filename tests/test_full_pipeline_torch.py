import unittest
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.vbx_segmenter import VoiceFemininityScoring

class TestPyTorchFullPipeline(unittest.TestCase):
    
    def test_execution(self):
        print("\nTesting Segmenter (PyTorch backend) on musanmix.mp3...")
        # vad_engine='smn' is default
        seg = Segmenter() 
        ret = seg('./media/musanmix.mp3')
        print(f"Result (first 3 segments): {ret[:3]}")
        self.assertTrue(len(ret) > 0)
        
    def test_short_file(self):
        print("\nTesting Segmenter (short file) on 0021.mp3...")
        seg = Segmenter(vad_engine='sm')
        ret = seg('./media/0021.mp3')
        ref = [('male', 0, 0.66)]
        print(f"Result: {ret}")
        self.assertEqual(ref, ret)

    def test_vfs(self):
        print("\nTesting VoiceFemininityScoring (PyTorch backend) on lamartine.wav...")
        media = './media/lamartine.wav'
        desired_vfp_score = 0.534884
        vfs = VoiceFemininityScoring(gd_model_criteria="vfp")
        score, dur, nb = vfs(media)
        print(f"VFS Score: {score}")
        
        np.testing.assert_almost_equal(
            score,
            desired_vfp_score,
            decimal=4,
            err_msg='Voice Femininity Score mismatch'
        )

if __name__ == '__main__':
    unittest.main()