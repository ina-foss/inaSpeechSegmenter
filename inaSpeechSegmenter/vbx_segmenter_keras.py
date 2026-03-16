
import os
import numpy as np
import logging

# Import Keras for legacy backend
import keras
from pyannote.core import Segment, Annotation, Timeline

from .features_vbx import povey_window, mel_fbank_mx, add_dither, fbank_htk, cmvn_floating_kaldi
from .segmenter_keras import Segmenter
from .io import media2sig16kmono
from .remote_utils import get_remote
from .vbx_segmenter import is_mid_speech, add_needed_vectors, get_femininity_score, get_annot_VAD, get_features, OnnxBackendExtractor

# Re-use shared functions from vbx_segmenter where logic is identical (VAD helpers, feature extraction)
# We only reimplement VoiceFemininityScoring to use Keras models

logger = logging.getLogger(__name__)

class VoiceFemininityScoring:
    """
    (Legacy Keras Version)
    Perform VBx features extraction and give a voice femininity score.
    """

    def __init__(self, gd_model_criteria="bgc", backend='onnx'):
        # VBx Extractor
        assert backend in ['onnx'], "Backend should be 'onnx'"
        if backend == "onnx":
            self.xvector_model = OnnxBackendExtractor()

        # Gender detection model
        assert gd_model_criteria in ["bgc", "vfp"], "Gender detection model Criteria must be 'bgc' (default) or 'vfp'"
        gd_model = None
        if gd_model_criteria == "bgc":
            gd_model = "interspeech2023_all.hdf5"
            self.vad_thresh = 0.7
        elif gd_model_criteria == "vfp":
            gd_model = "interspeech2023_cvfr.hdf5"
            self.vad_thresh = 0.62
            
        # Use Keras to load the model
        model_path = get_remote(gd_model)
        self.gender_detection_mlp_model = keras.models.load_model(model_path, compile=False)

        # Voice activity detection model (Legacy Keras Segmenter)
        self.vad = Segmenter(vad_engine='smn', detect_gender=False)

    def apply_vad(self, xvectors, a_vad):
        midpoint_seg = []
        n_xvectors = []
        for key, (start, stop), x in xvectors:
            if is_mid_speech(start, stop, a_vad):
                seg_total_duration = stop - start
                seg_cropped = Timeline([Segment(start, stop)]).crop(a_vad.get_timeline())
                if seg_cropped.duration() / seg_total_duration >= self.vad_thresh:
                    n_xvectors.append((key, (start, stop), x))
                midpoint_seg.append(((seg_cropped.duration() / seg_total_duration), key, Segment(start, stop), x))

        return add_needed_vectors(n_xvectors, midpoint_seg)

    def __call__(self, fpath):
        SR = 16000
        basename, _ = os.path.splitext(os.path.basename(fpath))[0], os.path.splitext(os.path.basename(fpath))[1]

        # Read "wav" file
        signal = media2sig16kmono(fpath, dtype="float64")
        duration = len(signal) / SR

        # Applying voice activity detection
        vad_seg = self.vad(fpath)
        annot_vad = get_annot_VAD(vad_seg)
        speech_duration = annot_vad.label_duration("speech")

        if speech_duration:
            # Processing features (mel bands extraction)
            features = get_features(signal)

            # Get xvector embeddings
            x_vectors = self.xvector_model(basename, features, duration)

            # VAD application (before gender detection)
            x_vectors = self.apply_vad(x_vectors, annot_vad)

            # Applying gender detection (Keras prediction)
            x = np.asarray([x for _, _, x in x_vectors])
            gender_pred = self.gender_detection_mlp_model.predict(x, verbose=0)
            if len(gender_pred) > 1:
                gender_pred = np.squeeze(gender_pred)

            gender_pred = np.asarray(
                [(segtup[0], segtup[1], pred) for (_, segtup, _), pred in zip(x_vectors, gender_pred)])

            score, nb_vectors = get_femininity_score(gender_pred), len(gender_pred)

        else:
            score, nb_vectors = None, 0

        return score, speech_duration, nb_vectors
