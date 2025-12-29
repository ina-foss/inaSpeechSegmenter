
import os
import sys
import shutil
import unittest
import numpy as np
import torch
import warnings
import gc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inaSpeechSegmenter.segmenter import DnnSegmenter, Segmenter, _get_patches, _binidx2seglist, _media2feats, _energy_activity
from inaSpeechSegmenter.vbx_segmenter import VoiceFemininityScoring, get_annot_VAD, get_features, get_femininity_score, is_mid_speech, add_needed_vectors
from inaSpeechSegmenter.pyannote_viterbi import viterbi_decoding
from inaSpeechSegmenter.viterbi_utils import diag_trans_exp
from inaSpeechSegmenter.pytorch_models import SpeechMusicCNN, SpeechMusicNoiseCNN, MaleFemaleCNN, Interspeech2023All, Interspeech2023Cvfr
from inaSpeechSegmenter.remote_utils import get_remote
from inaSpeechSegmenter.io import media2sig16kmono

# --- Patched Classes for PyTorch Support ---

class TorchDnnSegmenter(DnnSegmenter):
    def __init__(self, batch_size, model_cls, model_fname):
        self.model_fname = model_fname
        self.nmel = model_cls().conv1.in_channels if hasattr(model_cls(), 'conv1') else 24 # slight hack, or assume from subclass
        # self.viterbi_arg and self.outlabels will be set by subclass
        
        pth_fname = model_fname.replace('.hdf5', '.pth')
        if os.path.exists(pth_fname):
            pth_path = os.path.abspath(pth_fname)
        else:
            pth_path = get_remote(pth_fname)
            
        self.nn = model_cls()
        self.nn.load_state_dict(torch.load(pth_path, weights_only=True))
        self.nn.eval()
        self.batch_size = batch_size

    def __call__(self, mspec, lseg, difflen=0):
        if self.nmel < 24:
            mspec = mspec[:, :self.nmel].copy()
        
        patches, finite = _get_patches(mspec, 68, 2)
        if difflen > 0:
            patches = patches[:-int(difflen / 2), :, :]
            finite = finite[:-int(difflen / 2)]
            
        assert len(finite) == len(patches), (len(patches), len(finite)) 
            
        batch = []
        batch_indices = []
        for i, (lab, start, stop) in enumerate(lseg):
            if lab == self.inlabel:
                batch.append(patches[start:stop, :])
                batch_indices.append(i)

        rawpred = None
        if len(batch) > 0:
            # Keras input: (N, H, W, 1) -> PyTorch input: (N, 1, H, W)
            # patches is (TotalFrames, H, W)
            # batch concatenation makes it (BatchFrames, H, W)
            batch_np = np.concatenate(batch)
            # Add channel dim for Keras compatibility logic (H, W, 1) then transpose
            # But here we can just add channel dim at pos 1
            batch_torch = torch.from_numpy(batch_np).unsqueeze(1).float() # (BatchFrames, 1, H, W)
            
            # Run inference in batches
            preds_list = []
            with torch.no_grad():
                for i in range(0, len(batch_torch), self.batch_size):
                    b = batch_torch[i:i+self.batch_size]
                    p = self.nn(b)
                    preds_list.append(p.cpu().numpy())
            rawpred = np.concatenate(preds_list)
        
        gc.collect()
            
        ret = []
        # Reconstruct segmentation
        # We need to distribute rawpred back to the segments
        pred_idx = 0
        
        for lab, start, stop in lseg:
            if lab != self.inlabel:
                ret.append((lab, start, stop))
                continue

            l = stop - start
            r = rawpred[pred_idx : pred_idx+l]
            pred_idx += l
            
            # Set predictions for non-finite frames to uniform/uncertain
            # (Original code set to 0.5 for 2 classes? Or 1/n_classes?)
            # Original: r[finite[start:stop] == False, :] = 0.5 
            # Note: 0.5 implies 2 classes. For 3 classes (SMN) it might be different in principle,
            # but let's stick to original logic.
            r[finite[start:stop] == False, :] = 1.0 / len(self.outlabels)
            
            pred = viterbi_decoding(np.log(r + 1e-10), diag_trans_exp(self.viterbi_arg, len(self.outlabels)))
            for lab2, start2, stop2 in _binidx2seglist(pred):
                ret.append((self.outlabels[int(lab2)], start2+start, stop2+start))
        return ret

class TorchSpeechMusic(TorchDnnSegmenter):
    outlabels = ('speech', 'music')
    inlabel = 'energy'
    viterbi_arg = 150
    def __init__(self, batch_size):
        super().__init__(batch_size, SpeechMusicCNN, 'keras_speech_music_cnn.hdf5')
        self.nmel = 21

class TorchSpeechMusicNoise(TorchDnnSegmenter):
    outlabels = ('speech', 'music', 'noise')
    inlabel = 'energy'
    viterbi_arg = 80
    def __init__(self, batch_size):
        super().__init__(batch_size, SpeechMusicNoiseCNN, 'keras_speech_music_noise_cnn.hdf5')
        self.nmel = 21

class TorchGender(TorchDnnSegmenter):
    outlabels = ('female', 'male')
    inlabel = 'speech'
    viterbi_arg = 80
    def __init__(self, batch_size):
        super().__init__(batch_size, MaleFemaleCNN, 'keras_male_female_cnn.hdf5')
        self.nmel = 24

class TorchVoiceFemininityScoring(VoiceFemininityScoring):
    def __init__(self, gd_model_criteria="bgc", backend='onnx'):
        # Initialize parent just to set up VAD, but we'll overwrite the gender model
        # We pass backend='onnx' because we haven't ported xvector extraction yet (it's ONNX or Torch-ResNet)
        # We only want to swap the MLP part.
        super().__init__(gd_model_criteria, backend)
        
        if gd_model_criteria == "bgc":
            model_cls = Interspeech2023All
            model_name = "interspeech2023_all.hdf5"
        elif gd_model_criteria == "vfp":
            model_cls = Interspeech2023Cvfr
            model_name = "interspeech2023_cvfr.hdf5"
            
        pth_fname = model_name.replace('.hdf5', '.pth')
        if os.path.exists(pth_fname):
            pth_path = os.path.abspath(pth_fname)
        else:
            pth_path = get_remote(pth_fname)
            
        self.gender_detection_mlp_model = model_cls()
        self.gender_detection_mlp_model.load_state_dict(torch.load(pth_path, weights_only=True))
        self.gender_detection_mlp_model.eval()
        
        # Override VAD with Torch version
        self.vad = TorchSegmenter(vad_engine='smn', detect_gender=False)

    def __call__(self, fpath):
        # We mostly copy the original __call__ but change the prediction line
        # Or easier: monkeypatch the prediction method of the model?
        # No, let's copy the logic to be safe and explicit.
        
        SR = 16000
        basename, _ = os.path.splitext(os.path.basename(fpath))[0], os.path.splitext(os.path.basename(fpath))[1]
        signal = media2sig16kmono(fpath, dtype="float64")
        duration = len(signal) / SR

        # Torch VAD
        vad_seg = self.vad(fpath)
        annot_vad = get_annot_VAD(vad_seg)
        speech_duration = annot_vad.label_duration("speech")

        if speech_duration:
            features = get_features(signal)
            x_vectors = self.xvector_model(basename, features, duration)
            x_vectors = self.apply_vad(x_vectors, annot_vad)

            # --- PyTorch Inference Here ---
            x_data = np.asarray([x for _, _, x in x_vectors])
            x_torch = torch.from_numpy(x_data).float()
            
            with torch.no_grad():
                gender_pred = self.gender_detection_mlp_model(x_torch).cpu().numpy()
            # ------------------------------

            if len(gender_pred) > 1:
                gender_pred = np.squeeze(gender_pred)

            gender_pred = np.asarray(
                [(segtup[0], segtup[1], pred) for (_, segtup, _), pred in zip(x_vectors, gender_pred)])

            score, nb_vectors = get_femininity_score(gender_pred), len(gender_pred)

        else:
            score, nb_vectors = None, 0

        return score, speech_duration, nb_vectors


class TorchSegmenter(Segmenter):
    def __init__(self, vad_engine='smn', detect_gender=True, ffmpeg='ffmpeg', batch_size=32, energy_ratio=0.03):
        if ffmpeg is not None:
            if shutil.which(ffmpeg) is None:
                raise(Exception("""ffmpeg program not found"""))
        self.ffmpeg = ffmpeg
        self.energy_ratio = energy_ratio

        assert vad_engine in ['sm', 'smn']
        if vad_engine == 'sm':
            self.vad = TorchSpeechMusic(batch_size)
        elif vad_engine == 'smn':
            self.vad = TorchSpeechMusicNoise(batch_size)

        self.detect_gender = detect_gender
        if detect_gender:
            self.gender = TorchGender(batch_size)


# --- Test Case ---

class TestTorchInaSpeechSegmenter(unittest.TestCase):
    
    def test_execution_torch(self):
        print("\nTesting TorchSegmenter on musanmix.mp3...")
        seg = TorchSegmenter()
        ret = seg('./media/musanmix.mp3')
        print(f"Result (first 3 segments): {ret[:3]}")
        self.assertTrue(len(ret) > 0)
        
    def test_short_torch(self):
        print("\nTesting TorchSegmenter (short file) on 0021.mp3...")
        seg = TorchSegmenter(vad_engine='sm')
        ret = seg('./media/0021.mp3')
        ref = [('male', 0, 0.66)]
        print(f"Result: {ret}")
        self.assertEqual(ref, ret)

    def test_vfs_torch(self):
        print("\nTesting TorchVoiceFemininityScoring on lamartine.wav...")
        media = './media/lamartine.wav'
        # Tolerance might need adjustment slightly due to float precision diffs between TF/Torch
        desired_vfp_score = 0.534884
        vfs = TorchVoiceFemininityScoring(gd_model_criteria="vfp")
        score, dur, nb = vfs(media)
        print(f"VFS Score: {score}")
        
        # We expect close results to the original Keras version
        # Using decimal=4 for safety
        np.testing.assert_almost_equal(
            score,
            desired_vfp_score,
            decimal=4,
            err_msg='Voice Femininity Score mismatch (Torch vs Expected)'
        )

if __name__ == '__main__':
    unittest.main()
