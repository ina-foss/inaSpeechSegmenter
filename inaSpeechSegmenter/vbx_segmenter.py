import os
from abc import ABC, abstractmethod
import numpy as np
import onnxruntime as ort
import logging
# import torch.backends

import keras
from pyannote.core import Segment, Annotation, Timeline

# from .resnet import ResNet101
from .features_vbx import povey_window, mel_fbank_mx, add_dither, fbank_htk, cmvn_floating_kaldi
from .segmenter import Segmenter
from .io import media2sig16kmono
from .remote_utils import get_remote

# torch.backends.cudnn.enabled = True
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STEP = 24
WINLEN = 144
FEAT_DIM = 64
EMBED_DIM = 256
SR = 16000


def is_mid_speech(segments, vad_timeline):
    """
    Compute midpoint of segment and return speech detected segment (from Voice activity detection)
    """
    m_bool = np.array([vad_timeline.overlapping((start + end) / 2) != [] for start, end in segments])
    return np.array(segments)[m_bool.astype(bool)]


def add_needed_seg(segments, t_mid):
    """
    Keep at least 50% preds whose midpoint is in speech segment
    """
    min_pred = round(0.5 * len(t_mid))
    if len(segments) < min_pred:
        # Sort array descending
        t_mid = np.asarray(t_mid)
        t_mid = t_mid[t_mid[:, 0].argsort()][::-1]
        diff = min_pred - len(segments)
        for _, start, stop in t_mid[len(segments):len(segments) + diff]:
            segments.append((start, stop))
    return segments


def get_femininity_score(g_preds):
    a_temp = Annotation()
    for start, stop, p in g_preds:
        a_temp[Segment(start, stop), '_'] = (p >= 0.5)

    # Return binary score and number of retained predictions
    return len(a_temp.label_timeline(True)) / len(a_temp)


def get_timeline(vad_tuples):
    return Timeline(segments=[Segment(start, end) for lab, start, end in vad_tuples if lab == "speech"])


def get_features(signal, LC=150, RC=149):
    """
    This code function is entirely copied from the VBx script 'predict.py'
    https://github.com/BUTSpeechFIT/VBx/blob/master/VBx/predict.py
    """

    noverlap = 240
    winlen = 400
    window = povey_window(winlen)
    fbank_mx = mel_fbank_mx(
        winlen, SR, NUMCHANS=FEAT_DIM, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)

    np.random.seed(3)  # for reproducibility
    signal = add_dither((signal * 2 ** 15).astype(int))
    seg = np.r_[signal[noverlap // 2 - 1::-1], signal, signal[-1:-winlen // 2 - 1:-1]]
    fea = fbank_htk(seg, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
    fea = cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)
    return fea


def get_timecodes(flength, duration):
    """
    Return list of (seg_start, seg_end) based on len(features)
    """
    res = [(round(i / 100.0, 3), (round((i + WINLEN) / 100.0, 3))) for i in range(0, flength - WINLEN, STEP)]
    lstart_seg = res[-1][0]
    if flength - lstart_seg - STEP >= 10:
        res.append((round(lstart_seg + STEP / 100.0, 3), round(duration, 2)))
    return res


class VoiceFemininityScoring:
    """
    Perform VBx features extraction and give a voice femininity score.
    """

    def __init__(self, gd_model_criteria="bgc", backend='onnx'):
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
        self.vad = Segmenter(vad_engine='smn', detect_gender=False)

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

    def __call__(self, fpath, tmpdir=None):
        """
        Return Voice Femininity Score of a given file with values before last sigmoid activation :
                * convert file to wav 16k mono with ffmpeg
                * operate Mel bands extraction
                * operate voice activity detection using ISS VAD ('smn')
                * get VBx features on detected speech segments
                * apply gender detection model and compute femininity score
                * return score, duration of detected speech and number of retained x-vectors
        """
        # Read "wav" file
        signal = media2sig16kmono(fpath, tmpdir, dtype="float64")
        duration = len(signal) / SR

        # Applying voice activity detection
        vad_seg = self.vad(fpath)
        speech_timeline = get_timeline(vad_seg)

        if speech_timeline.duration():

            # Processing features (mel bands extraction)
            features = get_features(signal)

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

        else:
            score, nb_vectors = None, 0

        return score, speech_timeline.duration(), nb_vectors


class VBxExtractor(ABC):
    """
    VBxExtractor is an abstract class performing xvector extraction.
    """

    @abstractmethod
    def __init__(self):
        """
        Method to be implemented by each extractor.
        Initialize model according to the chosen backend.
        """
        pass

    def __call__(self, segments, features):
        xvectors = []
        for start, stop in segments:
            data = features[int(start * 100):int(stop * 100)]
            xvector = self.get_embedding(data)
            if np.isnan(xvector).any():
                logger.warning(f'NaN found, not processing: Segment({start}:{stop}) {os.linesep}')
            else:
                xvectors.append(((start, stop), xvector))

        # Multiply all vbx vectors by 10 (output standardization to get std=1)
        return [(seg, x * 10) for seg, x in xvectors]


class OnnxBackendExtractor(VBxExtractor):
    def __init__(self):
        model_path = get_remote("final.onnx")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            model = ort.InferenceSession(model_path, so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            model = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
        self.input_name = model.get_inputs()[0].name
        self.label_name = model.get_outputs()[0].name
        self.model = model

    def get_embedding(self, fea):
        return self.model.run(
            [self.label_name],
            {self.input_name: fea.astype(np.float32).transpose()[np.newaxis, :, :]}
        )[0].squeeze()

# # Backend implementation with torch
# # See VBx project : https://github.com/BUTSpeechFIT/VBx/blob/master/VBx/predict.py
#
# class TorchBackendExtractor(VBxExtractor):
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         model_path = get_remote("raw_81.pth")
#         model = ResNet101(feat_dim=FEAT_DIM, embed_dim=EMBED_DIM)
#         model = model.to(self.device)
#         checkpoint = torch.load(model_path, map_location=self.device)
#         model.load_state_dict(checkpoint['state_dict'], strict=False)
#         model.eval()
#         self.model = model
#
#     def get_embedding(self, fea):
#         with torch.no_grad():
#             data = torch.from_numpy(fea).to(self.device)
#             data = data[None, :, :]
#             data = torch.transpose(data, 1, 2)
#             spk_embeds = self.model(data)
#             return spk_embeds.data.cpu().numpy()[0]
