import os
from abc import ABC, abstractmethod
import logging

import numpy as np
import onnxruntime as ort
from pyannote.core import Segment, Annotation, Timeline
import keras

from .remote_utils import get_remote
from .features_vbx import povey_window, mel_fbank_mx, add_dither, fbank_htk, cmvn_floating_kaldi

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


def get_timecodes(flength, duration):
    """
    Return list of (seg_start, seg_end) based on len(features)
    """
    res = [(round(i / 100.0, 3), (round((i + WINLEN) / 100.0, 3))) for i in range(0, flength - WINLEN, STEP)]
    lstart_seg = res[-1][0]
    if flength - (lstart_seg*100) - STEP >= 10:
        res.append((round(lstart_seg + STEP / 100.0, 3), round(duration, 2)))
    return res


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
