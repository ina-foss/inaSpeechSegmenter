import os
import pandas as pd
import numpy as np
import onnxruntime as ort
import logging
import torch.backends

import keras
from pyannote.core import Segment, Annotation, Timeline

from .resnet import ResNet101
from .features_vbx import povey_window, mel_fbank_mx, add_dither, fbank_htk, cmvn_floating_kaldi
from .segmenter import Segmenter
from .io import media2sig16kmono
from .remote_utils import get_remote

torch.backends.cudnn.enabled = True
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STEP = 24
WINLEN = 144
FEAT_DIM = 64
EMBED_DIM = 256
SR = 16000


def annot_to_df(annotation):
    seg_tuples = [(s.start, s.end, label) for s, _, label in annotation.itertracks(yield_label=True)]
    df = pd.DataFrame.from_records(seg_tuples, columns=["start", "stop", "label"])
    return df


def is_mid_speech(start, stop, a_vad):
    """
    Compute midpoint of segment and return True if it's a speech detected segment (from Voice activity detection)
    """
    m = (start + stop) / 2
    is_speech = [True if seg.start < m < seg.end else False for seg, _, _ in a_vad.itertracks(yield_label=True)]
    return np.any(is_speech)


def get_binary_gender_label(p):
    return "female" if (p >= 0.5) else "male"


def add_needed_vectors(a_res, t_mid):
    min_pred = round(0.5 * len(t_mid))
    if len(a_res) < min_pred:
        # Sort array descending
        t_mid = np.asarray(t_mid)
        t_mid = t_mid[t_mid[:, 0].argsort()][::-1]
        diff = min_pred - len(a_res)
        for _, s, p in t_mid[len(a_res):len(a_res) + diff]:
            a_res[s, '_'] = get_binary_gender_label(p)

    return a_res


def get_femininity_score(g_pred, a_vad, overlap_tresh):
    res = Annotation()
    midpoint_seg = []

    for start, stop, pred in g_pred:

        # Keep segment label whose segment midpoint is in a speech segment
        if is_mid_speech(start, stop, a_vad):
            seg_total_duration = stop - start
            seg_cropped = Timeline([Segment(start, stop)]).crop(a_vad.get_timeline())
            # At least x % of the segment is detected as speech
            if seg_cropped.duration() / seg_total_duration >= overlap_tresh:
                res[Segment(start, stop), '_'] = get_binary_gender_label(pred)
            # Save overlap ratio with vad
            midpoint_seg.append(((seg_cropped.duration() / seg_total_duration), Segment(start, stop), pred))

    # Add vectors with vad-overlap if too many preds have been removed
    # Keep at least 50% preds whose midpoint is in speech segment
    res = add_needed_vectors(res, midpoint_seg)

    # Return binary score and number of retained predictions
    return len(res.label_timeline("female")) / len(res), len(res)


def get_annot_VAD(vad_tuples):
    annot_vad = Annotation()
    for lab, start, end in vad_tuples:
        if lab == "speech":
            annot_vad[Segment(start, end), '_'] = lab
    return annot_vad


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


class VoiceFemininityScoring:
    """
    Perform VBx features extraction and give a voice femininity score.
    """
    def __init__(self, backend='onnx', gd_model_criteria="bgc"):
        """
        Load VBx model weights according to the chosen backend
        (See : https://github.com/BUTSpeechFIT/VBx)
        Load Voice activity detection from inaSpeechSegmenter and finally
        load Gender detection model to estimate voice femininity.
        """

        # VBx Extractor
        assert backend in ['onnx', 'pytorch'], "Backend should be 'pytorch' or 'onnx'."
        if backend == "onnx":
            self.xvector_model = OnnxBackendExtractor()
        elif backend == "pytorch":
            self.xvector_model = TorchBackendExtractor()

        # Gender detection model
        assert gd_model_criteria in ["bgc", "vfp"], "Gender detection model Criteria must be 'bgc' (default) or 'vfp'"
        gd_model = None
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

    def __call__(self, fpath, tmpdir=None):
        """
        Return Voice Femininity Score of a given file with values before last sigmoid activation :
                * convert file to wav 16k mono with ffmpeg
                * operate Mel bands extraction
                * get VBx features
                * operate voice activity detection using ISS VAD ('smn')
                * apply gender detection model and compute femininity score
                * return score, duration of detected speech and number of retained x-vectors
        """
        basename, ext = os.path.splitext(os.path.basename(fpath))[0], os.path.splitext(os.path.basename(fpath))[1]

        # Read "wav" file
        signal = media2sig16kmono(fpath, tmpdir)
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

            # Applying gender detection (pretrained Multi layer perceptron)
            x = np.asarray([x * 10 for _, _, x in x_vectors])
            gender_pred = self.gender_detection_mlp_model.predict(x, verbose=0)
            if len(gender_pred) > 1:
                gender_pred = np.squeeze(gender_pred)

            # Link segment start/stop from x-vectors extraction to gender predictions
            gender_pred = np.asarray(
                [(segtup[0], segtup[1], pred) for (_, segtup, _), pred in zip(x_vectors, gender_pred)])

            score, nb_vectors = get_femininity_score(gender_pred, annot_vad, overlap_tresh=self.vad_thresh)

        else:
            score, nb_vectors = None, 0

        return score, speech_duration, nb_vectors


class VBxExtractor:
    """
    Extractor is an abstract class performing xvector extraction.
    """
    def __call__(self, basename, fea, duration):
        xvectors = []
        start = 0
        for start in range(0, len(fea) - WINLEN, STEP):
            data = fea[start:start + WINLEN]
            xvector = self.get_embedding(data)
            key = f'{basename}_{start:08}-{(start + WINLEN):08}'
            if np.isnan(xvector).any():
                logger.warning(f'NaN found, not processing: {key}{os.linesep}')
            else:
                seg_start = round(start / 100.0, 3)
                seg_end = round(start / 100.0 + WINLEN / 100.0, 3)
                xvectors.append((key, (seg_start, seg_end), xvector))

        #  Last segment
        if len(fea) - start - STEP >= 10:
            data = fea[start + STEP:len(fea)]
            xvector = self.get_embedding(data)
            key = f'{basename}_{(start + STEP):08}-{len(fea):08}'
            if np.isnan(xvector).any():
                logger.warning(f'NaN found, not processing: {key}{os.linesep}')
            else:
                seg_start = round((start + STEP) / 100.0, 3)
                seg_end = round(duration, 3)
                xvectors.append((key, (seg_start, seg_end), xvector))
        return xvectors


class OnnxBackendExtractor(VBxExtractor):
    # Class to perform VBx-based extraction when chosen backend is ONNX
    def __init__(self):
        model_path = get_remote("final.onnx")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        model = ort.InferenceSession(model_path, so, providers=["CUDAExecutionProvider"])
        self.device = None
        self.input_name = model.get_inputs()[0].name
        self.label_name = model.get_outputs()[0].name
        self.model = model

    def get_embedding(self, fea):
        return self.model.run(
            [self.label_name],
            {self.input_name: fea.astype(np.float32).transpose()[np.newaxis, :, :]}
        )[0].squeeze()


class TorchBackendExtractor(VBxExtractor):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = get_remote("raw_81.pth")
        model = ResNet101(feat_dim=FEAT_DIM, embed_dim=EMBED_DIM)
        model = model.to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()
        self.model = model

    def get_embedding(self, fea):
        with torch.no_grad():
            data = torch.from_numpy(fea).to(self.device)
            data = data[None, :, :]
            data = torch.transpose(data, 1, 2)
            spk_embeds = self.model(data)
            return spk_embeds.data.cpu().numpy()[0]

