#!python2.7

# ABBAS KHOSRAVANI
# INTELLIGENT VOICE
# http://intelligentvoice.com

"""
Speaker Diarization Using Recurrent Neural Networks
Usage:
  diarize [ --maxSpks=<maxSpks> \
            --CUDA_VISIBLE=<cuda> \
            --pcaDim=<pcaDim> \
            --nJobs=<nJobs> \
            --maxDur=<maxDur> \
            --fwin=<winlen> \
            --loopProb=<prob> \
            --minDur=<minDur> \
            --output=<output>] <wav_file> <psrt_file> <model_path>
  diarize -h | --help
  diarize --version
Options:
  <wav_file>                 Set path to the audio file.
                             It should be an 8kHz 16-bits audio file.

  <psrt_file>                Set path to the transcription file.
  
  <model_path>               Set path to the model.
                             The model should be in h5 format. It could be generator_latest.h5.
                             
  --maxSpks=<maxSpks>        Maximun of speakers present in the audio file [default: 2]

  --CUDA_VISIBLE=<cuda>      The CUDA_VISIBLE_DEVICES [default: 1]

  --pcaDim=<pcaDim>         The PCA dimension used prior to clustering [default: 5]

  --nJobs=<nJobs>           Number of jobs used by spectral clustering [default: 8]

  --maxDur=<maxDur>          The maximum segment duration in second used to split long audio files and process in parallel.
                             [default: 10]

  --loopProb=<prob>          Probability of not switching speakers between frames [default: 0.99].

  --minDur=<minDur>          Minimum number of frames between speaker turns imposed by linear
                             chains of HMM states corresponding to each speaker. All the states
                             in a chain share the same output distribution [default: 10].

  --output=<spkfile>         Path to the outout speaker time tags file.
                             [defualt: the same as the wav_file but with .spk extension]
                             The .spk file is formated as [stime dur spk]
                             
  -h --help                  Show this screen.
  --version                  Show version.

"""

import numpy as np
import os
import pickle
import librosa
from scipy.stats import mode
import scipy.sparse as sparse
from spectral import SpectralClustering 
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

from docopt import docopt

# Perform diarization
def diarization(
    feat,
    sad,
    spkfile,
    generator=None,
    filename=None,
    loopProb=0.99,
    minDur=1,
    maxSpks=10,
    pcaDim=8,
    nJobs=-1,
    maxDur=600,
    batch_size=128):
    
    n_dim = feat.shape[-1]

    # feat = feat[sad]
    X = []
    step = maxDur / 6
    for i in range(0, len(feat) - maxDur + 1, step):
        X.append(feat[i:i + maxDur])

    x = np.zeros_like(feat[0:maxDur])
    x[:len(feat[i + step:])] = feat[i + step:]
    X[-1] = x

    X = generator.predict(np.asarray(X), batch_size=batch_size)

    # print(calinski_harabaz_score(Z_t, labels))
    # print(silhouette_score(Z_t, labels))

    pca = PCA(n_components=pcaDim, copy=True, whiten=True)
    Z = pca.fit_transform(X)

    scores = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(Z)
        scores.append(calinski_harabaz_score(Z, kmeans.labels_))
    
    maxSpks = np.argmax(scores) + 2
    print("Estimated number of speakers: " + str(maxSpks))

    cluster = SpectralClustering(
        n_clusters=maxSpks,
        eigen_solver='arpack',
        # affinity="rbf",
        affinity="nearest_neighbors",
        n_init=10,
        n_neighbors=10,
        assign_labels='discretize',
        n_jobs=nJobs).fit(Z)


    labels = cluster.labels_    
    cluster.prob_[cluster.prob_ <= 1e-3] = 1e-3
    lls = np.log(cluster.prob_)
    sp = np.asarray([np.sum(labels == s, dtype=float) for s in np.unique(labels)]) / len(labels)
    tr = np.eye(minDur * maxSpks, k=1)
    tr[minDur-1::minDur, 0::minDur] = (1 - loopProb) * sp
    tr[(np.arange(1, maxSpks + 1) * minDur - 1,) * 2] += loopProb
    q, tll, lf, lb = forward_backward(
        lls.repeat(minDur, axis=1),
        tr,
        tr[::-1].diagonal(),
        np.arange(1, maxSpks + 1) * minDur - 1)

    q = q.reshape(len(q), maxSpks, minDur).sum(axis=2)
    labels = np.argmax(q, -1) + 1
    
    spk2lab = np.zeros(len(sad))
    q = spk2lab
    idx = 0
    for i in range(0, len(feat) - maxDur + 1, step):
        q[i:i + step] = labels[idx]
        idx += 1
    q[i + step:] = labels[-1]
    spk2lab = q

    return spk2lab

def logsumexp(x, axis=0):
    xmax = x.max(axis)
    x = xmax + np.log(np.sum(np.exp(x - np.expand_dims(xmax, axis)), axis))
    infs = np.isinf(xmax)
    if np.ndim(x) > 0:
      x[infs] = xmax[infs]
    elif infs:
      x = xmax
    return x

def forward_backward(lls, tr, ip, fs):
    """
    Inputs:
        lls - matrix of per-frame log HMM state output probabilities 
        tr  - transition probability matrix
        ip  - vector of initial state probabilities (i.e. statrting in the state)
        fs  - vector of indices of final states
    Outputs:
        sp  - matrix of per-frame state occupation posteriors 
        tll - total (forward) log-likelihood
        lfw - log forward probabilities
        lfw - log backward probabilities
    """ 
    ltr = np.log(tr + 1e-5)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls) 
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    lfw[0] = lls[0] + np.log(ip + 1e-5)
    lbw[-1] = 0.0
    
    for ii in  xrange(1,len(lls)):
        lfw[ii] =  lls[ii] + logsumexp(lfw[ii-1] + ltr.T, axis=1)
  
    for ii in reversed(xrange(len(lls)-1)):
        lbw[ii] = logsumexp(ltr + lls[ii+1] + lbw[ii+1], axis=1)
    
    tll = logsumexp(lfw[-1])
    sp = np.exp(lfw + lbw - tll)
    return sp, tll, lfw, lbw

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Speaker Diarization (RNN) v1.0')
    wavfile = os.path.abspath(arguments['<wav_file>'])
    psrtfile = os.path.abspath(arguments['<psrt_file>'])
    model_path = os.path.abspath(arguments['<model_path>'])

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    spkfile = wavfile.replace('.wav', '.spk')

    if arguments['--output']:
        spkfile = os.path.abspath(arguments['--output'])
    if arguments['--CUDA_VISIBLE']:
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments['--CUDA_VISIBLE']
    if arguments['--maxSpks']:
        maxSpks = int(arguments['--maxSpks'])
    if arguments['--maxDur']:
        maxDur = int(arguments['--maxDur']) * 100
    if arguments['--loopProb']:
        loopProb = float(arguments['--loopProb'])
    if arguments['--minDur']:
        minDur = int(arguments['--minDur'])
    if arguments['--nJobs']:
        nJobs = int(arguments['--nJobs'])
    if arguments['--pcaDim']:
        pcaDim = int(arguments['--pcaDim'])
    if arguments['--fwin']:
        fwin = int(arguments['--fwin'])

    # loading the speaker embedding generator
    from keras.models import load_model
    generator = load_model(model_path)

    # performing the diarization
    print('diarizing ...')
    q = diarization(
        wavfile,
        psrtfile,
        spkfile,
        maxDur=maxDur,
        pcaDim=pcaDim,
        nJobs=nJobs,
        minDur=minDur,
        loopProb=loopProb,
        generator=generator,
        maxSpks=maxSpks)
