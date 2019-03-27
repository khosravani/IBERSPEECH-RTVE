
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

wavList = set(glob.glob('/home/abbas/abbas/workspace/iberspeech/dev2/audio/*.wav'))

print('preparing a reference dataset ...')
reference = {}
for f in tqdm(wavList):
    wavfile = os.path.abspath(f)
    fname = wavfile.split('/')[-1].split('.')[0]
    ref = Annotation(uri=fname)
    reference[fname] = ref
    tags = np.loadtxt(
        wavfile.replace('/audio/', '/spk/').replace('.wav', '.spk'),
        dtype={'names': ('start', 'end', 'spk'), 'formats': ('f8', 'f8', 'S10')})
    
    for r in tags.tolist():
        ref[Segment( r[0], r[1] )] = r[2]

feats_path = '/home/abbas/abbas/workspace/data/sre16/v2/data/iberspeech_dev2/feats.scp'
fd = open_or_fd(feats_path)
feats = {}
try:
    for line in fd:
        key, rxfile = line.decode().split(' ')
        feats[key] = read_mat(rxfile)
finally:
    if fd is not feats_path : fd.close()

vad_path = '/home/abbas/abbas/workspace/data/sre16/v2/data/iberspeech_dev2/vad.scp'
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

    if fname == "LN24H-20151125":
        continue

    spkpath = 'dev2/hyp/' + fname + '.rttm'

    lab = np.loadtxt(
        'dev2/spk/' + fname + '.spk',
        dtype={
            'names': ('start', 'end', 'spk'), 
            'formats': ('f8', 'f8', 'S10')
        })
    N = np.sum(lab['end'] - lab['start'])
    # for spk in np.unique(lab['spk']):
    #     print(spk, '%.2f' % (np.sum(lab['end'][lab['spk']==spk] - lab['start'][[lab['spk']==spk]]) / N * 100))

    print("Total number of speakers: " + str(len(np.unique(lab['spk']))))
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
                    ref[Segment(idx, ((i - 1) * 10 + 25) / 1000.)] = state
                idx = ((i - 1) * 10 + 25) / 1000.
                state = s

    print('{0}: DER = {1:.2f}%'.format(fname, 100. * metric(reference[fname], hypothesis[fname])))
    print('{0}: Confusion = {1:.2f}s'.format(fname, metric['confusion']))
    print('{0}: Correct = {1:.2f}s'.format(fname, metric['correct']))
    print('{0}: Total = {1:.2f}s'.format(fname, metric['total']))
    print('{0}: False Alarm = {1:.2f}s'.format(fname, metric['false alarm']))
    print('{0}: Missed Detection = {1:.2f}s\n'.format(fname, metric['missed detection']))
    print('Total DER = {0:.2f}%\n'.format(100 * abs(metric)))

