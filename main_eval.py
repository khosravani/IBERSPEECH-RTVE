
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from tqdm import tqdm
import os
import glob
from kaldi_io import open_or_fd, read_mat, read_vec_flt
from pyannote.core import Segment, Timeline, Annotation, notebook
from pyannote.metrics.diarization import DiarizationErrorRate

from diarize import diarization
from keras.models import load_model
import h5py
import pickle

np.set_printoptions(threshold=np.nan)

generator = load_model('../models/generator_cnn_kaldi_174.h5')

wavList = set(glob.glob('/home/abbas/abbas/workspace/iberspeech/test/audio/*.wav'))

feats_path = '/home/abbas/abbas/workspace/data/sre16/v2/data/iberspeech_test/feats.scp'
fd = open_or_fd(feats_path)
feats = {}
try:
    for line in fd:
        key, rxfile = line.decode().split(' ')
        feats[key] = read_mat(rxfile)
finally:
    if fd is not feats_path : fd.close()

vad_path = '/home/abbas/abbas/workspace/data/sre16/v2/data/iberspeech_test/vad.scp'
fd = open_or_fd(vad_path)
vads = {}
try:
    for line in fd:
        key, rxfile = line.decode().split(' ')
        vads[key] = read_vec_flt(rxfile).astype(bool)
finally:
    if fd is not vad_path : fd.close()

print('diarizing the test dataset ...')
hypothesis = {}
metric = DiarizationErrorRate(collar=0.250, skip_overlap=True)
for f in tqdm(feats):
    fname = f[5:-2]
    spkpath = 'test/rttm/' + fname + '.rttm'

    ref = Annotation(uri=fname)
    hypothesis[fname] = ref
    q = diarization(
        feats[f],
        vads[f],
        spkpath,
        generator=generator)
    
    idx = 0.
    state = q[0]
    with open(spkpath, 'w') as f:
        for i, s in enumerate(q):
            if s != state:
                if state != 0:
                    f.write("SPEAKER %s 1 %.2f %.2f <NA> <NA> #_%d <NA> <NA>\n" % (fname, idx, ((i - 1) * 10 + 25) / 1000. - idx, state))
                idx = ((i - 1) * 10 + 25) / 1000.
                state = s

