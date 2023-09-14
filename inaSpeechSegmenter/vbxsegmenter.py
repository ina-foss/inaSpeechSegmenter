import numpy as np
import keras
from librosa.sequence import transition_loop, viterbi

from .utils import OnnxBackendExtractor, get_timeline, binidx2seglist
from .remote_utils import get_remote

WINLEN = 144
STEP = 24
VITERBI_PR = 0.99999994


def get_indices(segments, vad_timeline):
    """
    Compute midpoint of segment and return indices of speech detected segment (from Voice activity detection)
    """
    m_bool = np.array([vad_timeline.overlapping((start + end) / 2) != [] for start, end in segments])
    ind = np.where(m_bool)[0]

    # The first speech segment is too short to find a midpoint (.<720ms)
    if not np.any(ind) and (vad_timeline[0].start == 0.0):
        ind = [0, 1]

    return ind[0], ind[-1] + 1


def get_timecodes(flength):
    """
    Return list of (seg_start, seg_end) based on len(features)
    Remark : This method is different from the ones used in VoiceFemininityScoring
    """

    last_start = 0

    res = [(round(i / 100.0, 3), (round((i + WINLEN) / 100.0, 3))) for i in range(0, flength - WINLEN, STEP)]
    
    if len(res) != 0:
        last_start = res[-1][0]

    # Add last segments (as long as segment duration is > 100 ms)
    while flength - (last_start * 100) - STEP > 10:
        res.append((round(last_start + STEP / 100.0, 3), round(flength / 100, 2)))
        last_start = res[-1][0]

    return res


class VBxSegmenter:
    def __init__(self):
        """
        Load VBx model weights for gender detection
        (See : https://github.com/BUTSpeechFIT/VBx)
        """

        # VBx Extractor - Onnx Backend
        model_name = "interspeech2023_all.hdf5"
        self.xvector_model = OnnxBackendExtractor()
        self.gender_detection_mlp_model = keras.models.load_model(get_remote(model_name), compile=False)
        self.inlabel = "speech"

        # Parameters for features extraction
        self.outlabels = ('male', 'female')

    def __call__(self, feats, lseg):


        mspec = feats.mspec_vbx        

        # Convert in sec (in future should be done outside this method)
        lseg = [(lab, start * 0.02, stop * 0.02) for lab, start, stop in lseg]

        # Get timecodes of each segment
        vbx_segments = get_timecodes(len(mspec))

        # For each speech segment
        ret = []
        for lab, start, stop in lseg:
            if lab != self.inlabel:
                ret.append((lab, start, stop))
                continue

            # Keep segment label whose segment midpoint is in a speech segment
            i, j = get_indices(vbx_segments, get_timeline([(lab, start, stop)]))

            # Get xvector (entire file)
            x_vectors = self.xvector_model(vbx_segments[i:j], mspec)
            x = np.asarray([x for _, x in x_vectors])

            # Apply gender detection
            gender_pred = self.gender_detection_mlp_model.predict(x, verbose=0)

            if len(gender_pred) > 1:
                gender_pred = np.squeeze(gender_pred)

            # Viterbi (librosa)
            r = np.vstack([1 - gender_pred, gender_pred])
            pred = viterbi(r, transition_loop(2, prob=VITERBI_PR))

            for lab2, start2, stop2 in binidx2seglist(pred):
                start2, stop2 = round((STEP*0.01) * start2, 2), round((STEP*0.01) * stop2, 2)
                ret.append((self.outlabels[int(lab2)], start + start2, start + stop2))

            # The stop of last gender segment is replaced by the stop of the speech segment
            llabel, lstart, _ = ret.pop()
            ret.append((llabel, lstart, stop))

        return [(lab, start / .02, stop / .02) for lab, start, stop in ret]
