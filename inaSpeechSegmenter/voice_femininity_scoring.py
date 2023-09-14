import numpy as np

import keras
from pyannote.core import Segment, Timeline


from .segmenter import Segmenter
from .remote_utils import get_remote
from .utils import OnnxBackendExtractor, is_mid_speech, add_needed_seg, get_timecodes, get_timeline, get_femininity_score

# torch.backends.cudnn.enabled = True

from .media2feats import CpuFeatExtractor


class VoiceFemininityScoring:
    """
    Perform VBx features extraction and give a voice femininity score.
    """

    def __init__(self, gd_model_criteria="bgc", backend='onnx', ffmpeg='ffmpeg', tmpdir=None):
        """
        Load VBx model weights according to the chosen backend
        (See : https://github.com/BUTSpeechFIT/VBx)
        Load Voice activity detection from inaSpeechSegmenter and finally
        load Gender detection model to estimate voice femininity.
        """

        # VBx Extractor
        assert backend in ['onnx'], "Backend should be 'onnx' (or 'pytorch' if uncommented)."
        if backend == "onnx":
            self.xvector_model = OnnxBackendExtractor()
        # elif backend == "pytorch":
        #     self.xvector_model = TorchBackendExtractor()

        # Gender detection model
        assert gd_model_criteria in ["bgc", "vfp"], f"""
        Gender detection model criteria must be 'bgc' (default) or 'vfp'. Provided criteria : {gd_model_criteria}
        """
        if gd_model_criteria == "bgc":
            gd_model = "interspeech2023_all.hdf5"
            self.vad_thresh = 0.7
        elif gd_model_criteria == "vfp":
            gd_model = "interspeech2023_cvfr.hdf5"
            self.vad_thresh = 0.62
        self.gender_detection_mlp_model = keras.models.load_model(
            get_remote(gd_model),
            compile=False)

        # Voice activity detection model
        self.vad = Segmenter(vad_engine='smn', gender_engine=None, ffmpeg=ffmpeg, tmpdir=tmpdir)
        self.feat_extractor = CpuFeatExtractor(True, True, ffmpeg, tmpdir)


    def apply_vad(self, segments, timeline):
        res, midpoint_seg = [], []

        # Keep segment label whose segment midpoint is in a speech segment
        retained_seg = is_mid_speech(segments, timeline)

        for start, stop in retained_seg:
            sdur = stop - start
            seg_cropped = Timeline([Segment(start, stop)]).crop(timeline)
            # At least x % of the segment is detected as speech
            if seg_cropped.duration() / sdur >= self.vad_thresh:
                res.append((start, stop))
            # Save overlap ratio with vad
            midpoint_seg.append((seg_cropped.duration() / sdur, start, stop))

        # Add segments with vad-overlap if too many predictions have been removed
        return add_needed_seg(res, midpoint_seg)

    def __call__(self, fpath):
        """
        Return Voice Femininity Score of a given file with values before last sigmoid activation :
                * convert file to wav 16k mono with ffmpeg
                * operate Mel bands extraction
                * operate voice activity detection using ISS VAD ('smn')
                * get VBx features on detected speech segments
                * apply gender detection model and compute femininity score
                * return score, duration of detected speech and number of retained x-vectors
        """

        # extract features & decode sound file
        feats = self.feat_extractor(fpath, None, None)

        # Applying voice activity detection
        vad_seg = self.vad.segment_feats(feats)
        # convert to pyannote annotation and keep only speech segments
        speech_timeline = get_timeline(vad_seg)

        # if no speech is detected, return
        if not speech_timeline.duration():
            return None, speech_timeline.duration(), 0

        # we use vbx mel bands
        features = feats.mspec_vbx
        duration = feats.duration

        # VAD application
        segments = get_timecodes(len(features), duration)
        retained_seg = self.apply_vad(segments, speech_timeline)

        # Get xvector embeddings
        x_vectors = self.xvector_model(retained_seg, features)

        # Applying gender detection (pretrained Multi layer perceptron)
        x = np.asarray([x for _, x in x_vectors])
        gender_pred = self.gender_detection_mlp_model.predict(x, verbose=0)
        if len(gender_pred) > 1:
            gender_pred = np.squeeze(gender_pred)

        # Link segment start/stop from x-vectors extraction to gender predictions
        gender_pred = np.asarray(
            [(segtup[0], segtup[1], pred) for (segtup, _), pred in zip(x_vectors, gender_pred)])

        score, nb_vectors = get_femininity_score(gender_pred), len(gender_pred)



        return score, speech_timeline.duration(), nb_vectors
