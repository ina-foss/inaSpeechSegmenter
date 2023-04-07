import os
import pandas as pd
import numpy as np
import tempfile
from subprocess import Popen, PIPE
import onnxruntime as ort
import logging
import torch.backends
import soundfile as sf
import h5py

import keras
from keras.utils import get_file
from pyannote.core import Segment, Annotation

from inaSpeechSegmenter.resnet import ResNet101
import inaSpeechSegmenter.features_vbx as ft
from inaSpeechSegmenter.segmenter import Segmenter

torch.backends.cudnn.enabled = True
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def initialize_gpus(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def get_embedding(fea, model, device, label_name=None, input_name=None, backend='pytorch'):
    if backend == 'pytorch':
        data = torch.from_numpy(fea).to(device)
        data = data[None, :, :]
        data = torch.transpose(data, 1, 2)
        spk_embeds = model(data)
        return spk_embeds.data.cpu().numpy()[0]
    elif backend == 'onnx':
        return model.run([label_name],
                         {input_name: fea.astype(np.float32).transpose()
                         [np.newaxis, :, :]})[0].squeeze()


@torch.no_grad()
def xvector_extraction(model, basename, fea, start, slen, winlen, step, label_name, input_name, duration, backend,
                       device, save_seg=False):
    xvectors = []
    content = []

    for start in range(0, slen - winlen, step):
        data = fea[start:start + winlen]
        xvector = get_embedding(data, model, device, label_name=label_name, input_name=input_name, backend=backend)
        key = f'{basename}_{start:08}-{(start + winlen):08}'
        if np.isnan(xvector).any():
            logger.warning(f'NaN found, not processing: {key}{os.linesep}')
        else:
            seg_start = round(start / 100.0, 3)
            seg_end = round(start / 100.0 + winlen / 100.0, 3)
            if save_seg:
                content.append(f'{key} {basename} {seg_start} {seg_end}{os.linesep}')
            xvectors.append((key, (seg_start, seg_end), xvector))

    #  Last segment
    if slen - start - step >= 10:
        data = fea[start + step:slen]
        xvector = get_embedding(data, model, device, label_name=label_name, input_name=input_name, backend=backend)
        key = f'{basename}_{(start + step):08}-{slen:08}'
        if np.isnan(xvector).any():
            logger.warning(f'NaN found, not processing: {key}{os.linesep}')
        else:
            seg_start = round((start + step) / 100.0, 3)
            seg_end = round(duration, 3)
            if save_seg:
                content.append(f'{key} {basename} {seg_start} {seg_end}{os.linesep}')
            xvectors.append((key, (seg_start, seg_end), xvector))

    if save_seg:
        f = open(basename + ".seg", "w")
        for c in content:
            f.write(c)
        f.close()

    return xvectors


def save_embeddings(tuple_xvec, name):
    d = {k: vector for k, vector in tuple_xvec}
    with h5py.File(f'{name}_xvectors.hdf5', 'w') as f:
        for key in d:
            f.create_dataset(key, data=d[key])


def annot_to_df(annotation):
    seg_tuples = [(s.start, s.end, label) for s, _, label in annotation.itertracks(yield_label=True)]
    df = pd.DataFrame.from_records(seg_tuples, columns=["start", "stop", "label"])
    return df


def get_femininity_score(g_pred, a_vad):
    # Get middle of prediction segments
    mid_lab_tuples = [((start + stop) / 2, "female" if (pred >= 0.5) else "male") for start, stop, pred in
                      g_pred]
    df_mid_seg = pd.DataFrame.from_records(mid_lab_tuples, columns=["mid", "lab"])

    # Keep segment label whose segment midpoint is in a speech segment
    res = [l for seg, _, _ in a_vad.itertracks(yield_label=True) for m, l in zip(df_mid_seg['mid'], df_mid_seg['lab'])
           if seg.start < m < seg.end]
    assert set(res) == {"female", "male"}, "Associated prediction labels are not in ['female', 'male']"
    n_female = res.count("female")

    return n_female / len(res), len(res)


def get_annot_VAD(vad_tuples):
    annot_vad = Annotation()
    for lab, start, end in vad_tuples:
        if lab == "speech":
            annot_vad[Segment(start, end), '_'] = lab
    return annot_vad


def launch_ffmpeg(medianame, tmpdir, start_sec=None, stop_sec=None):
    base, _ = os.path.splitext(os.path.basename(medianame))

    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:
        # build ffmpeg command line
        tmpwav = tmpdirname + '/' + base + '.wav'
        args = ['ffmpeg', '-y', '-i', medianame, '-ar', '16000', '-ac', '1']
        if start_sec is None:
            start_sec = 0
        else:
            args += ['-ss', '%f' % start_sec]

        if stop_sec is not None:
            args += ['-to', '%f' % stop_sec]
        args += [tmpwav]

        # launch ffmpeg
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        output, error = p.communicate()
        assert p.returncode == 0, error

        sig, sr = sf.read(f'{tmpwav}')
    dur = len(sig) / sr

    return sig, sr, dur


class VoiceFemininityScoring:
    def __init__(self, backend='onnx', gpu='', save_segments=False, gd_model_criteria="bgc"):
        """
        Load VBx model weights
        """
        url = 'https://github.com/ina-foss/inaSpeechSegmenter/releases/download/interspeech23/'

        assert backend in ['onnx', 'pytorch'], "Backend should be 'pytorch' or 'onnx'."
        self.backend = backend
        self.step = 24
        self.winlen = 144
        self.feat_dim = 64
        self.embed_dim = 256
        self.save_segments = save_segments

        if gpu != '':
            logger.info(f'Using GPU: {gpu}')
            # gpu configuration
            initialize_gpus(gpu)
            self.device = torch.device(device='cuda')
        else:
            self.device = torch.device(device='cpu')

        model, label_name, input_name = '', None, None

        if backend == 'pytorch':
            if gpu == '':
                print("""
If you want to use a GPU with backend='pytorch', specify it in the initialization parameters
Current chosen device : %s
                """ % self.device)
            model_path = get_file("raw_81.pth", url + "raw_81.pth", cache_dir="interspeech23")
            model = ResNet101(feat_dim=self.feat_dim, embed_dim=self.embed_dim)
            model = model.to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
            self.input_name = None
            self.label_name = None
        elif backend == 'onnx':
            model_path = get_file("final.onnx", url + "final.onnx", cache_dir="interspeech23")
            so = ort.SessionOptions()
            so.log_severity_level = 3
            model = ort.InferenceSession(model_path, so, providers=["CUDAExecutionProvider"])
            input_name = model.get_inputs()[0].name
            label_name = model.get_outputs()[0].name
            self.input_name = input_name
            self.label_name = label_name

        self.xvector_model = model
        assert gd_model_criteria in ["bgc", "vfp"], "Gender detection model Criteria must be 'bgc' (default) or 'vfp'"
        gd_model = None
        if gd_model_criteria == "bgc":
            gd_model = "interspeech2023_all.hdf5"
        elif gd_model_criteria == "vfp":
            gd_model = "interspeech2023_cvfr.hdf5"
        self.gender_detection_mlp_model = keras.models.load_model(
            get_file(gd_model, url + gd_model, cache_dir="interspeech23"),
            compile=False)

        self.vad = Segmenter(vad_engine='smn', detect_gender=False)

    def get_features(self, signal, sr, LC=150, RC=149):
        """
        This code function is entirely copied from the VBx script 'predict.py'
        Input :
            - fpath : path of ".wav" file
        """

        if sr == 16000:
            noverlap = 240
            winlen = 400
            window = ft.povey_window(winlen)
            fbank_mx = ft.mel_fbank_mx(
                winlen, sr, NUMCHANS=self.feat_dim, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
        else:
            raise ValueError(f'Only 16kHz is supported. Got {sr} instead.')

        np.random.seed(3)  # for reproducibility
        signal = ft.add_dither((signal * 2 ** 15).astype(int))
        seg = np.r_[signal[noverlap // 2 - 1::-1], signal, signal[-1:-winlen // 2 - 1:-1]]
        fea = ft.fbank_htk(seg, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        fea = ft.cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)
        slen = len(fea)
        start = -self.step

        return fea, slen, start

    def __call__(self, fpath, tmpdir=None):
        """
        Return Voice Femininity Score of a given file with values before last sigmoid activation :
                * convert file to wav 16k mono with ffmpeg
                * operate Mel bands extraction
                * get VBx features
                * operate voice activity detection using ISS VAD ('smn')
                * apply gender detection model and compute femininity score
                * apply gender detection model without applying last sigmoid activation
                * return score, duration of detected speech and number of retained x-vectors
        """
        basename, ext = os.path.splitext(os.path.basename(fpath))[0], os.path.splitext(os.path.basename(fpath))[1]

        # Read "wav" file
        signal, samplerate, duration = launch_ffmpeg(fpath, tmpdir)

        # process segment only if longer than 0.01s
        if signal.shape[0] > 0.01 * samplerate:

            # Processing features (mel bands extraction)
            features, slen, start = self.get_features(signal, samplerate)

            # Get xvector embeddings
            x_vectors = xvector_extraction(self.xvector_model, basename, features, start, slen, self.winlen,
                                           self.step, self.label_name, self.input_name, duration,
                                           self.backend, device=self.device, save_seg=self.save_segments)
            # Applying voice activity detection
            vad_seg = self.vad(fpath)
            annot_vad = get_annot_VAD(vad_seg)
            speech_duration = annot_vad.label_duration("speech")

            # Applying gender detection (pretrained Multi layer perceptron)
            x = np.asarray([x * 10 for _, _, x in x_vectors])
            gender_pred = self.gender_detection_mlp_model.predict(x, verbose=0)
            if len(gender_pred) > 1:
                gender_pred = np.squeeze(gender_pred)

            # Link segment start/stop from x-vectors extraction to gender predictions
            gender_pred = np.asarray(
                [(segtup[0], segtup[1], pred) for (_, segtup, _), pred in zip(x_vectors, gender_pred)])

            # Femininity score (from binary predictions)
            if speech_duration:
                score, nb_xvector = get_femininity_score(gender_pred, annot_vad)
            else:
                score, nb_xvector = None, None

            return score, speech_duration, nb_xvector
