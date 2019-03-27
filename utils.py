import numpy
import numpy.matlib
import copy
import pandas
import wave
import struct
import os
import math
import ctypes
import multiprocessing
import warnings

import scipy
from scipy import ndimage
import scipy.stats as stats
from scipy.fftpack import fft
from scipy.signal import decimate
from scipy.signal import lfilter
from scipy.fftpack.realtransforms import dct

def read_sph(input_file_name, mode='p'):
    """
    Read a SPHERE audio file

    :param input_file_name: name of the file to read
    :param mode: specifies the following (\* =default)
    
    .. note::
    
        - Scaling:
        
            - 's'    Auto scale to make data peak = +-1 (use with caution if reading in chunks)
            - 'r'    Raw unscaled data (integer values)
            - 'p'    Scaled to make +-1 equal full scale
            - 'o'    Scale to bin centre rather than bin edge (e.g. 127 rather than 127.5 for 8 bit values,
                     can be combined with n+p,r,s modes)
            - 'n'    Scale to negative peak rather than positive peak (e.g. 128.5 rather than 127.5 for 8 bit values,
                     can be combined with o+p,r,s modes)

        - Format
       
           - 'l'    Little endian data (Intel,DEC) (overrides indication in file)
           - 'b'    Big endian data (non Intel/DEC) (overrides indication in file)

       - File I/O
       
           - 'f'    Do not close file on exit
           - 'd'    Look in data directory: voicebox('dir_data')
           - 'w'    Also read the annotation file \*.wrd if present (as in TIMIT)
           - 't'    Also read the phonetic transcription file \*.phn if present (as in TIMIT)

        - NMAX     maximum number of samples to read (or -1 for unlimited [default])
        - NSKIP    number of samples to skip from start of file (or -1 to continue from previous read when FFX
                   is given instead of FILENAME [default])

    :return: a tupple such that (Y, FS)
    
    .. note::
    
        - Y data matrix of dimension (samples,channels)
        - FS         sample frequency in Hz
        - WRD{\*,2}  cell array with word annotations: WRD{\*,:)={[t_start t_end],'text'} where times are in seconds
                     only present if 'w' option is given
        - PHN{\*,2}  cell array with phoneme annotations: PHN{\*,:)={[t_start	t_end],'phoneme'} where times
                     are in seconds only present if 't' option is present
        - FFX        Cell array containing

            1. filename
            2. header information
        
            1. first header field name
            2. first header field value
            3. format string (e.g. NIST_1A)
            4. 
                1. file id
                2. current position in file
                3. dataoff    byte offset in file to start of data
                4. order  byte order (l or b)
                5. nsamp    number of samples
                6. number of channels
                7. nbytes    bytes per data value
                8. bits    number of bits of precision
                9. fs	sample frequency
                10. min value
                11. max value
                12. coding 0=PCM,1=uLAW + 0=no compression, 0=shorten,20=wavpack,30=shortpack
                13. file not yet decompressed
                
            5. temporary filename

    If no output parameters are specified,
    header information will be printed.
    The code to decode shorten-encoded files, is 
    not yet released with this toolkit.
    """
    codings = dict([('pcm', 1), ('ulaw', 2)])
    compressions = dict([(',embedded-shorten-', 1),
                         (',embedded-wavpack-', 2),
                         (',embedded-shortpack-', 3)])
    byteorder = 'l'
    endianess = dict([('l', '<'), ('b', '>')])

    if not mode == 'p':
        mode = [mode, 'p']
    k = list((m >= 'p') & (m <= 's') for m in mode)
    # scale to input limits not output limits
    mno = all([m != 'o' for m in mode])
    sc = ''
    if k[0]:
        sc = mode[0]
    # Get byte order (little/big endian)
    if any([m == 'l' for m in mode]):
        byteorder = 'l'
    elif any([m == 'b' for m in mode]):
        byteorder = 'b'
    ffx = ['', '', '', '', '']

    if isinstance(input_file_name, str):
        if os.path.exists(input_file_name):
            fid = open(input_file_name, 'rb')
        elif os.path.exists("".join((input_file_name, '.sph'))):
            input_file_name = "".join((input_file_name, '.sph'))
            fid = open(input_file_name, 'rb')
        else:
            raise Exception('Cannot find file {}'.format(input_file_name))
        ffx[0] = input_file_name
    elif not isinstance(input_file_name, str):
        ffx = input_file_name
    else:
        fid = input_file_name

    # Read the header
    if ffx[3] == '':
        fid.seek(0, 0)  # go to the begining of the file
        l1 = fid.readline().decode("utf-8")
        l2 = fid.readline().decode("utf-8")
        if not (l1 == 'NIST_1A\n') & (l2 == '   1024\n'):
            logging.warning('File does not begin with a SPHERE header')
        ffx[2] = l1.rstrip()
        hlen = int(l2[3:7])
        hdr = {}
        while True:  # Read the header and fill a dictionary
            st = fid.readline().decode("utf-8").rstrip()
            if st[0] != ';':
                elt = st.split(' ')
                if elt[0] == 'end_head':
                    break
                if elt[1][0] != '-':
                    logging.warning('Missing ''-'' in SPHERE header')
                    break
                if elt[1][1] == 's':
                    hdr[elt[0]] = elt[2]
                elif elt[1][1] == 'i':
                    hdr[elt[0]] = int(elt[2])
                else:
                    hdr[elt[0]] = float(elt[2])

        if 'sample_byte_format' in list(hdr.keys()):
            if hdr['sample_byte_format'][0] == '0':
                bord = 'l'
            else:
                bord = 'b'
            if (bord != byteorder) & all([m != 'b' for m in mode]) \
                    & all([m != 'l' for m in mode]):
                byteorder = bord

        icode = 0  # Get encoding, default is PCM
        if 'sample_coding' in list(hdr.keys()):
            icode = -1  # unknown code
            for coding in list(codings.keys()):
                if hdr['sample_coding'].startswith(coding):
                    # is the signal compressed
                    # if len(hdr['sample_coding']) > codings[coding]:
                    if len(hdr['sample_coding']) > len(coding):
                        for compression in list(compressions.keys()):
                            if hdr['sample_coding'].endswith(compression):
                                icode = 10 * compressions[compression] \
                                        + codings[coding] - 1
                                break
                    else:  # if the signal is not compressed
                        icode = codings[coding] - 1
                        break
        # initialize info of the files with default values
        info = [fid, 0, hlen, ord(byteorder), 0, 1, 2, 16, 1, 1, -1, icode]
        # Get existing info from the header
        if 'sample_count' in list(hdr.keys()):
            info[4] = hdr['sample_count']
        if not info[4]:  # if no info sample_count or zero
            # go to the end of the file
            fid.seek(0, 2)  # Go to te end of the file
            # get the sample count
            info[4] = int(math.floor((fid.tell() - info[2]) / (info[5] * info[6])))  # get the sample_count
        if 'channel_count' in list(hdr.keys()):
            info[5] = hdr['channel_count']
        if 'sample_n_bytes' in list(hdr.keys()):
            info[6] = hdr['sample_n_bytes']
        if 'sample_sig_bits' in list(hdr.keys()):
            info[7] = hdr['sample_sig_bits']
        if 'sample_rate' in list(hdr.keys()):
            info[8] = hdr['sample_rate']
        if 'sample_min' in list(hdr.keys()):
            info[9] = hdr['sample_min']
        if 'sample_max' in list(hdr.keys()):
            info[10] = hdr['sample_max']

        ffx[1] = hdr
        ffx[3] = info
    info = ffx[3]
    ksamples = info[4]
    if ksamples > 0:
        fid = info[0]
        if (icode >= 10) & (ffx[4] == ''):  # read compressed signal
            # need to use a script with SHORTEN
            raise Exception('compressed signal, need to unpack in a script with SHORTEN')
        info[1] = ksamples
        # use modes o and n to determine effective peak
        pk = 2 ** (8 * info[6] - 1) * (1 + (float(mno) / 2 - int(all([m != 'b'
                                                                      for m in
                                                                      mode]))) / 2 **
                                       info[7])
        fid.seek(1024)  # jump after the header
        nsamples = info[5] * ksamples
        if info[6] < 3:
            if info[6] < 2:
                logging.debug('Sphere i1 PCM')
                y = numpy.fromfile(fid, endianess[byteorder]+"i1", -1)
                if info[11] % 10 == 1:
                    if y.shape[0] % 2:
                        y = numpy.frombuffer(audioop.ulaw2lin(
                                numpy.concatenate((y, numpy.zeros(1, 'int8'))), 2),
                                numpy.int16)[:-1]/32768.
                    else:
                        y = numpy.frombuffer(audioop.ulaw2lin(y, 2), numpy.int16)/32768.
                    pk = 1.
                else:
                    y = y - 128
            else:
                logging.debug('Sphere i2')
                y = numpy.fromfile(fid, endianess[byteorder]+"i2", -1)
        else:  # non verifie
            if info[6] < 4:
                y = numpy.fromfile(fid, endianess[byteorder]+"i1", -1)
                y = y.reshape(nsamples, 3).transpose()
                y = (numpy.dot(numpy.array([1, 256, 65536]), y) - (numpy.dot(y[2, :], 2 ** (-7)).astype(int) * 2 ** 24))
            else:
                y = numpy.fromfile(fid, endianess[byteorder]+"i4", -1)

        if sc != 'r':
            if sc == 's':
                if info[9] > info[10]:
                    info[9] = numpy.min(y)
                    info[10] = numpy.max(y)
                sf = 1 / numpy.max(list(list(map(abs, info[9:11]))), axis=0)
            else:
                sf = 1 / pk
            y = sf * y

        if info[5] > 1:
            y = y.reshape(ksamples, info[5])
    else:
        y = numpy.array([])
    if mode != 'f':
        fid.close()
        info[0] = -1
        if not ffx[4] == '':
            pass  # VERIFY SCRIPT, WHICH CASE IS HANDLED HERE
    return y.astype(numpy.float32), int(info[8]), int(info[6])


def read_wav(input_file_name):
    """
    :param input_file_name:
    :return:
    """
    wfh = wave.open(input_file_name, "r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wfh.getparams()
    raw = wfh.readframes(nframes * nchannels)
    out = struct.unpack_from("%dh" % nframes * nchannels, raw)
    sig = numpy.reshape(numpy.array(out), (-1, nchannels)).squeeze()
    wfh.close()
    return sig.astype(numpy.float32), framerate, sampwidth


def read_pcm(input_file_name):
    """Read signal from single channel PCM 16 bits

    :param input_file_name: name of the PCM file to read.
    
    :return: the audio signal read from the file in a ndarray encoded  on 16 bits, None and 2 (depth of the encoding in bytes)
    """
    with open(input_file_name, 'rb') as f:
        f.seek(0, 2)  # Go to te end of the file
        # get the sample count
        sample_count = int(f.tell() / 2)
        f.seek(0, 0)  # got to the begining of the file
        data = numpy.asarray(struct.unpack('<' + 'h' * sample_count, f.read()))
    return data.astype(numpy.float32), None, 2


def read_audio(input_file_name, framerate=None):
    """ Read a 1 or 2-channel audio file in SPHERE, WAVE or RAW PCM format.
    The format is determined from the file extension.
    If the sample rate read from the file is a multiple of the one given
    as parameter, we apply a decimation function to subsample the signal.
    
    :param input_file_name: name of the file to read from
    :param framerate: frame rate, optional, if lower than the one read from the file, subsampling is applied
    :return: the signal as a numpy array and the sampling frequency
    """
    if framerate is None:
        raise TypeError("Expected sampling frequency required in sidekit.frontend.io.read_audio")
    ext = os.path.splitext(input_file_name)[-1]
    if ext.lower() == '.sph':
        sig, read_framerate, sampwidth = read_sph(input_file_name, 'p')
    elif ext.lower() == '.wav' or ext.lower() == '.wave':
        sig, read_framerate, sampwidth = read_wav(input_file_name)
    elif ext.lower() == '.pcm' or ext.lower() == '.raw':
        sig, read_framerate, sampwidth = read_pcm(input_file_name)
        read_framerate = framerate
    else:
        raise TypeError("Unknown extension of audio file")

    # Convert to 16 bit encoding if needed
    sig *= (2**(15-sampwidth))

    if framerate > read_framerate:
        print("Warning in read_audio, up-sampling function is not implemented yet!")
    elif read_framerate % float(framerate) == 0 and not framerate == read_framerate:
        print("downsample")
        sig = decimate(sig, int(read_framerate / float(framerate)), n=None, ftype='iir', axis=0)
    return sig.astype(numpy.float32), framerate


def rasta_filt(x):
    """Apply RASTA filtering to the input signal.
    
    :param x: the input audio signal to filter.
        cols of x = critical bands, rows of x = frame
        same for y but after filtering
        default filter is single pole at 0.94
    """
    x = x.T
    numerator = numpy.arange(.2, -.3, -.1)
    denominator = numpy.array([1, -0.94])

    # Initialize the state.  This avoids a big spike at the beginning
    # resulting from the dc offset level in each band.
    # (this is effectively what rasta/rasta_filt.c does).
    # Because Matlab uses a DF2Trans implementation, we have to
    # specify the FIR part to get the state right (but not the IIR part)
    y = numpy.zeros(x.shape)
    zf = numpy.zeros((x.shape[0], 4))
    for i in range(y.shape[0]):
        y[i, :4], zf[i, :4] = lfilter(numerator, 1, x[i, :4], axis=-1, zi=[0, 0, 0, 0])
    
    # .. but don't keep any of these values, just output zero at the beginning
    y = numpy.zeros(x.shape)

    # Apply the full filter to the rest of the signal, append it
    for i in range(y.shape[0]):
        y[i, 4:] = lfilter(numerator, denominator, x[i, 4:], axis=-1, zi=zf[i, :])[0]
    
    return y.T


def cms(features, label=None, global_mean=None):
    """Performs cepstral mean subtraction
    
    :param features: a feature stream of dimension dim x nframes 
            where dim is the dimension of the acoustic features and nframes the 
            number of frames in the stream
    :param label: a logical vector
    :param global_mean: pre-computed mean to use for feature normalization if given

    :return: a feature stream
    """
    # If no label file as input: all speech are speech
    if label is None:
        label = numpy.ones(features.shape[0]).astype(bool)
    if label.sum() == 0:
        mu = numpy.zeros((features.shape[1]))
    if global_mean is not None:
        mu = global_mean
    else:
        mu = numpy.mean(features[label, :], axis=0)
    features -= mu


def cmvn(features, label=None, global_mean=None, global_std=None):
    """Performs mean and variance normalization
    
    :param features: a feature stream of dimension dim x nframes 
        where dim is the dimension of the acoustic features and nframes the 
        number of frames in the stream
    :param global_mean: pre-computed mean to use for feature normalization if given
    :param global_std: pre-computed standard deviation to use for feature normalization if given
    :param label: a logical verctor

    :return: a sequence of features
    """
    # If no label file as input: all speech are speech
    if label is None:
        label = numpy.ones(features.shape[0]).astype(bool)

    if global_mean is not None and global_std is not None:
        mu = global_mean
        stdev = global_std
        features -= mu
        features /= stdev

    elif not label.sum() == 0:
        mu = numpy.mean(features[label, :], axis=0)
        stdev = numpy.std(features[label, :], axis=0)
        features -= mu
        features /= stdev


def stg(features, label=None, win=301):
    """Performs feature warping on a sliding window
    
    :param features: a feature stream of dimension dim x nframes 
        where dim is the dimension of the acoustic features and nframes the
        number of frames in the stream
    :param label: label of selected frames to compute the Short Term Gaussianization, by default, al frames are used
    :param win: size of the frame window to consider, must be an odd number to get a symetric context on left and right
    :return: a sequence of features
    """

    # If no label file as input: all speech are speech
    if label is None:
        label = numpy.ones(features.shape[0]).astype(bool)
    speech_features = features[label, :]

    add_a_feature = False
    if win % 2 == 1:
        # one feature per line
        nframes, dim = numpy.shape(speech_features)

        # If the number of frames is not enough for one window
        if nframes < win:
            # if the number of frames is not odd, duplicate the last frame
            # if nframes % 2 == 1:
            if not nframes % 2 == 1:
                nframes += 1
                add_a_feature = True
                speech_features = numpy.concatenate((speech_features, [speech_features[-1, ]]))
            win = nframes

        # create the output feature stream
        stg_features = numpy.zeros(numpy.shape(speech_features))

        # Process first window
        r = numpy.argsort(speech_features[:win, ], axis=0)
        r = numpy.argsort(r, axis=0)
        arg = (r[: (win - 1) / 2] + 0.5) / win
        stg_features[: (win - 1) / 2, :] = stats.norm.ppf(arg, 0, 1)

        # process all following windows except the last one
        for m in range(int((win - 1) / 2), int(nframes - (win - 1) / 2)):
            idx = list(range(int(m - (win - 1) / 2), int(m + (win - 1) / 2 + 1)))
            foo = speech_features[idx, :]
            r = numpy.sum(foo < foo[(win - 1) / 2], axis=0) + 1
            arg = (r - 0.5) / win
            stg_features[m, :] = stats.norm.ppf(arg, 0, 1)

        # Process the last window
        r = numpy.argsort(speech_features[list(range(nframes - win, nframes)), ], axis=0)
        r = numpy.argsort(r, axis=0)
        arg = (r[(win + 1) / 2: win, :] + 0.5) / win
        
        stg_features[list(range(int(nframes - (win - 1) / 2), nframes)), ] = stats.norm.ppf(arg, 0, 1)
    else:
        # Raise an exception
        raise Exception('Sliding window should have an odd length')

    # wrapFeatures = np.copy(features)
    if add_a_feature:
        stg_features = stg_features[:-1]
    features[label, :] = stg_features


def cep_sliding_norm(features, win=301, label=None, center=True, reduce=False):
    """
    Performs a cepstal mean substitution and standard deviation normalization
    in a sliding windows. MFCC is modified.

    :param features: the MFCC, a numpy array
    :param win: the size of the sliding windows
    :param label: vad label if available
    :param center: performs mean subtraction
    :param reduce: performs standard deviation division

    """
    if label is None:
        label = numpy.ones(features.shape[0]).astype(bool)

    if numpy.sum(label) <= win:
        if reduce:
            cmvn(features, label)
        else:
            cms(features, label)
    else:
        d_win = win // 2
        df = pandas.DataFrame(features[label, :])
        r = df.rolling(window=win, center=True)
        mean = r.mean().values
        std = r.std().values

        mean[0:d_win, :] = mean[d_win, :]
        mean[-d_win:, :] = mean[-d_win-1, :]

        std[0:d_win, :] = std[d_win, :]
        std[-d_win:, :] = std[-d_win-1, :]

        if center:
            features[label, :] -= mean
            if reduce:
                features[label, :] /= std


def pre_emphasis(input_sig, pre):
    """Pre-emphasis of an audio signal.
    :param input_sig: the input vector of signal to pre emphasize
    :param pre: value that defines the pre-emphasis filter. 
    """
    if input_sig.ndim == 1:
        return (input_sig - numpy.c_[input_sig[numpy.newaxis, :][..., :1],
                                     input_sig[numpy.newaxis, :][..., :-1]].squeeze() * pre)
    else:
        return input_sig - numpy.c_[input_sig[..., :1], input_sig[..., :-1]] * pre


    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    This method has been implemented by Anne Archibald, 
    as part of the talk box toolkit
    example::
    
        segment_axis(arange(10), 4, 2)
        array([[0, 1, 2, 3],
           ( [2, 3, 4, 5],
             [4, 5, 6, 7],
             [6, 7, 8, 9]])

    :param a: the array to segment
    :param length: the length of each frame
    :param overlap: the number of array elements by which the frames should overlap
    :param axis: the axis to operate on; if None, act on the flattened array
    :param end: what to do with the last frame, if the array is not evenly 
            divisible into pieces. Options are:
            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value

    :param endvalue: the value to use for end='pad'

    :return: a ndarray

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """

    if axis is None:
        a = numpy.ravel(a)  # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise ValueError("overlap must be nonnegative and length must" +
                         "be positive")

    if l < length or (l - length) % (length - overlap):
        if l > length:
            roundup = length + (1 + (l - length) // (length - overlap)) * (length - overlap)
            rounddown = length + ((l - length) // (length - overlap)) * (length - overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length - overlap) or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1, axis)

        if end == 'cut':
            a = a[..., :rounddown]
            l = a.shape[0]
        elif end in ['pad', 'wrap']:  # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = numpy.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup - l]
            a = b

        a = a.swapaxes(-1, axis)

    if l == 0:
        raise ValueError("Not enough data points to segment array " +
                         "in 'cut' mode; try 'pad' or 'wrap'")
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)
    s = a.strides[axis]
    new_shape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
    new_strides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]

    try:
        return numpy.ndarray.__new__(numpy.ndarray, strides=new_strides,
                                     shape=new_shape, buffer=a, dtype=a.dtype)
    except TypeError:
        a = a.copy()
        # Shape doesn't change but strides does
        new_strides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]
        return numpy.ndarray.__new__(numpy.ndarray, strides=new_strides,
                                     shape=new_shape, buffer=a, dtype=a.dtype)


def speech_enhancement(X, Gain, NN):
    """This program is only to process the single file seperated by the silence
    section if the silence section is detected, then a counter to number of
    buffer is set and pre-processing is required.

    Usage: SpeechENhance(wavefilename, Gain, Noise_floor)

    :param X: input audio signal
    :param Gain: default value is 0.9, suggestion range 0.6 to 1.4,
            higher value means more subtraction or noise redcution
    :param NN:
    
    :return: a 1-dimensional array of boolean that 
        is True for high energy frames.
    
    Copyright 2014 Sun Han Wu and Anthony Larcher
    """
    if X.shape[0] < 512:  # creer une exception
        return X

    num1 = 40  # dsiable buffer number
    Alpha = 0.75  # original value is 0.9
    FrameSize = 32 * 2  # 256*2
    FrameShift = int(FrameSize / NN)  # FrameSize/2=128
    nfft = FrameSize  # = FrameSize
    Fmax = int(numpy.floor(nfft / 2) + 1)  # 128+1 = 129
    # arising hamming windows
    Hamm = 1.08 * (0.54 - 0.46 * numpy.cos(2 * numpy.pi * numpy.arange(FrameSize) / (FrameSize - 1)))
    y0 = numpy.zeros(FrameSize - FrameShift)  # 128 zeros

    Eabsn = numpy.zeros(Fmax)
    Eta1 = Eabsn

    ###################################################################
    # initial parameter for noise min
    mb = numpy.ones((1 + FrameSize // 2, 4)) * FrameSize / 2  # 129x4  set four buffer * FrameSize/2
    im = 0
    Beta1 = 0.9024  # seems that small value is better;
    pxn = numpy.zeros(1 + FrameSize // 2)  # 1+FrameSize/2=129 zeros vector

    ###################################################################
    old_absx = Eabsn
    x = numpy.zeros(FrameSize)
    x[FrameSize - FrameShift:FrameSize] = X[
        numpy.arange(numpy.min((int(FrameShift), X.shape[0])))]

    if x.shape[0] < FrameSize:
        EOF = 1
        return X

    EOF = 0
    Frame = 0

    ###################################################################
    # add the pre-noise estimates
    for i in range(200):
        Frame += 1
        fftn = fft(x * Hamm)  # get its spectrum
        absn = numpy.abs(fftn[0:Fmax])  # get its amplitude

        # add the following part from noise estimation algorithm
        pxn = Beta1 * pxn + (1 - Beta1) * absn  # Beta=0.9231 recursive pxn
        im = (im + 1) % 40  # noise_memory=47;  im=0 (init) for noise level estimation

        if im:
            mb[:, 0] = numpy.minimum(mb[:, 0], pxn)  # 129 by 4 im<>0  update the first vector from PXN
        else:
            mb[:, 1:] = mb[:, :3]  # im==0 every 47 time shift pxn to first vector of mb
            mb[:, 0] = pxn
            #  0-2  vector shifted to 1 to 3

        pn = 2 * numpy.min(mb, axis=1)  # pn = 129x1po(9)=1.5 noise level estimate compensation
        # over_sub_noise= oversubtraction factor

        # end of noise detection algotihm
        x[:FrameSize - FrameShift] = x[FrameShift:FrameSize]
        index1 = numpy.arange(FrameShift * Frame, numpy.min((FrameShift * (Frame + 1), X.shape[0])))
        In_data = X[index1]  # fread(ifp, FrameShift, 'short');

        if In_data.shape[0] < FrameShift:  # to check file is out
            EOF = 1
            break
        else:
            x[FrameSize - FrameShift:FrameSize] = In_data  # shift new 128 to position 129 to FrameSize location
            # end of for loop for noise estimation

    # end of prenoise estimation ************************
    x = numpy.zeros(FrameSize)
    x[FrameSize - FrameShift:FrameSize] = X[numpy.arange(numpy.min((int(FrameShift), X.shape[0])))]

    if x.shape[0] < FrameSize:
        EOF = 1
        return X

    EOF = 0
    Frame = 0

    X1 = numpy.zeros(X.shape)
    Frame = 0

    while EOF == 0:
        Frame += 1
        xwin = x * Hamm

        fftx = fft(xwin, nfft)  # FrameSize FFT
        absx = numpy.abs(fftx[0:Fmax])  # Fmax=129,get amplitude of x
        argx = fftx[:Fmax] / (absx + numpy.spacing(1))  # normalize x spectrum phase

        absn = absx

        # add the following part from rainer algorithm
        pxn = Beta1 * pxn + (1 - Beta1) * absn  # s Beta=0.9231   recursive pxn

        im = int((im + 1) % (num1 * NN / 2))  # original =40 noise_memory=47;  im=0 (init) for noise level estimation

        if im:
            mb[:, 0] = numpy.minimum(mb[:, 0], pxn)  # 129 by 4 im<>0  update the first vector from PXN
        else:
            mb[:, 1:] = mb[:, :3]  # im==0 every 47 time shift pxn to first vector of mb
            mb[:, 0] = pxn

        pn = 2 * numpy.min(mb, axis=1)  # pn = 129x1po(9)=1.5 noise level estimate compensation

        Eabsn = pn
        Gaina = Gain

        temp1 = Eabsn * Gaina

        Eta1 = Alpha * old_absx + (1 - Alpha) * numpy.maximum(absx - temp1, 0)
        new_absx = (absx * Eta1) / (Eta1 + temp1)  # wiener filter
        old_absx = new_absx

        ffty = new_absx * argx  # multiply amplitude with its normalized spectrum

        y = numpy.real(numpy.fft.fftpack.ifft(numpy.concatenate((ffty,
                                                                 numpy.conj(ffty[numpy.arange(Fmax - 2, 0, -1)])))))

        y[:FrameSize - FrameShift] = y[:FrameSize - FrameShift] + y0
        y0 = y[FrameShift:FrameSize]  # keep 129 to FrameSize point samples 
        x[:FrameSize - FrameShift] = x[FrameShift:FrameSize]

        index1 = numpy.arange(FrameShift * Frame, numpy.min((FrameShift * (Frame + 1), X.shape[0])))
        In_data = X[index1]  # fread(ifp, FrameShift, 'short');

        z = 2 / NN * y[:FrameShift]  # left channel is the original signal 
        z /= 1.15
        z = numpy.minimum(z, 32767)
        z = numpy.maximum(z, -32768)
        index0 = numpy.arange(FrameShift * (Frame - 1), FrameShift * Frame)
        if not all(index0 < X1.shape[0]):
            idx = 0
            while (index0[idx] < X1.shape[0]) & (idx < index0.shape[0]):
                X1[index0[idx]] = z[idx]
                idx += 1
        else:
            X1[index0] = z

        if In_data.shape[0] == 0:
            EOF = 1
        else:
            x[numpy.arange(FrameSize - FrameShift, FrameSize + In_data.shape[0] - FrameShift)] = In_data

    X1 = X1[X1.shape[0] - X.shape[0]:]
    # }
    # catch{

    # }
    return X1


def vad_percentil(log_energy, percent):
    """

    :param log_energy:
    :param percent:
    :return:
    """
    thr = numpy.percentile(log_energy, percent)
    return log_energy > thr, thr


def vad_energy(log_energy,
               distrib_nb=3,
               nb_train_it=8,
               flooring=0.0001, ceiling=1.0,
               alpha=2):
    # center and normalize the energy
    log_energy = (log_energy - numpy.mean(log_energy)) / numpy.std(log_energy)

    # Initialize a Mixture with 2 or 3 distributions
    world = Mixture()
    # set the covariance of each component to 1.0 and the mean to mu + meanIncrement
    world.cst = numpy.ones(distrib_nb) / (numpy.pi / 2.0)
    world.det = numpy.ones(distrib_nb)
    world.mu = -2 + 4.0 * numpy.arange(distrib_nb) / (distrib_nb - 1)
    world.mu = world.mu[:, numpy.newaxis]
    world.invcov = numpy.ones((distrib_nb, 1))
    # set equal weights for each component
    world.w = numpy.ones(distrib_nb) / distrib_nb
    world.cov_var_ctl = copy.deepcopy(world.invcov)

    # Initialize the accumulator
    accum = copy.deepcopy(world)

    # Perform nbTrainIt iterations of EM
    for it in range(nb_train_it):
        accum._reset()
        # E-step
        world._expectation(accum, log_energy)
        # M-step
        world._maximization(accum, ceiling, flooring)

    # Compute threshold
    threshold = world.mu.max() - alpha * numpy.sqrt(1.0 / world.invcov[world.mu.argmax(), 0])

    # Apply frame selection with the current threshold
    label = log_energy > threshold
    return label, threshold


def vad_snr(sig, snr, fs=16000, shift=0.01, nwin=256):
    """Select high energy frames based on the Signal to Noise Ratio
    of the signal.
    Input signal is expected encoded on 16 bits
    
    :param sig: the input audio signal
    :param snr: Signal to noise ratio to consider
    :param fs: sampling frequency of the input signal in Hz. Default is 16000.
    :param shift: shift between two frames in seconds. Default is 0.01
    :param nwin: number of samples of the sliding window. Default is 256.
    """
    overlap = nwin - int(shift * fs)
    sig /= 32768.
    sig = speech_enhancement(numpy.squeeze(sig), 1.2, 2)
    
    # Compute Standard deviation
    sig += 0.1 * numpy.random.randn(sig.shape[0])
    
    std2 = segment_axis(sig, nwin, overlap, axis=None, end='cut', endvalue=0).T
    std2 = numpy.std(std2, axis=0)
    std2 = 20 * numpy.log10(std2)  # convert the dB

    # APPLY VAD
    label = (std2 > numpy.max(std2) - snr) & (std2 > -75)

    return label


def label_fusion(label, win=3):
    """Apply a morphological filtering on the label to remove isolated labels.
    In case the input is a two channel label (2D ndarray of boolean of same 
    length) the labels of two channels are fused to remove
    overlaping segments of speech.
    
    :param label: input labels given in a 1D or 2D ndarray
    :param win: parameter or the morphological filters
    """
    channel_nb = len(label)
    if channel_nb == 2:
        overlap_label = numpy.logical_and(label[0], label[1])
        label[0] = numpy.logical_and(label[0], ~overlap_label)
        label[1] = numpy.logical_and(label[1], ~overlap_label)

    for idx, lbl in enumerate(label):
        cl = ndimage.grey_closing(lbl, size=win)
        label[idx] = ndimage.grey_opening(cl, size=win)

    return label


def hz2mel(f, htk=True):
    """Convert an array of frequency in Hz into mel.
    
    :param f: frequency to convert
    
    :return: the equivalence on the mel scale.
    """
    if htk:
        return 2595 * numpy.log10(1 + f / 700.)
    else:
        f = numpy.array(f)

        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        f_0 = 0.
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = f < brkfrq

        z = numpy.zeros_like(f)
        # fill in parts separately
        z[linpts] = (f[linpts] - f_0) / f_sp
        z[~linpts] = brkpt + (numpy.log(f[~linpts] / brkfrq)) / numpy.log(logstep)

        if z.shape == (1,):
            return z[0]
        else:
            return z


def mel2hz(z, htk=True):
    """Convert an array of mel values in Hz.
    
    :param m: ndarray of frequencies to convert in Hz.
    
    :return: the equivalent values in Hertz.
    """
    if htk:
        return 700. * (10**(z / 2595.) - 1)
    else:
        z = numpy.array(z, dtype=float)
        f_0 = 0
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = (z < brkpt)

        f = numpy.zeros_like(z)

        # fill in parts separately
        f[linpts] = f_0 + f_sp * z[linpts]
        f[~linpts] = brkfrq * numpy.exp(numpy.log(logstep) * (z[~linpts] - brkpt))

        if f.shape == (1,):
            return f[0]
        else:
            return f


def hz2bark(f):
    """
    Convert frequencies (Hertz) to Bark frequencies

    :param f: the input frequency
    :return:
    """
    return 6. * numpy.arcsinh(f / 600.)


def bark2hz(z):
    """
    Converts frequencies Bark to Hertz (Hz)

    :param z:
    :return:
    """
    return 600. * numpy.sinh(z / 6.)


def compute_delta(features,
                  win=3,
                  method='filter',
                  filt=numpy.array([.25, .5, .25, 0, -.25, -.5, -.25])):
    """features is a 2D-ndarray  each row of features is a a frame
    
    :param features: the feature frames to compute the delta coefficients
    :param win: parameter that set the length of the computation window.
            The size of the window is (win x 2) + 1
    :param method: method used to compute the delta coefficients
        can be diff or filter
    :param filt: definition of the filter to use in "filter" mode, default one
        is similar to SPRO4:  filt=numpy.array([.2, .1, 0, -.1, -.2])
        
    :return: the delta coefficients computed on the original features.
    """
    # First and last features are appended to the begining and the end of the 
    # stream to avoid border effect
    x = numpy.zeros((features.shape[0] + 2 * win, features.shape[1]), dtype=numpy.float32)
    x[:win, :] = features[0, :]
    x[win:-win, :] = features
    x[-win:, :] = features[-1, :]

    delta = numpy.zeros(x.shape, dtype=numpy.float32)

    if method == 'diff':
        filt = numpy.zeros(2 * win + 1, dtype=numpy.float32)
        filt[0] = -1
        filt[-1] = 1

    for i in range(features.shape[1]):
        delta[:, i] = numpy.convolve(features[:, i], filt)

    return delta[win:-win, :]


def pca_dct(cep, left_ctx=12, right_ctx=12, p=None):
    """Apply DCT PCA as in [McLaren 2015] paper:
    Mitchell McLaren and Yun Lei, 'Improved Speaker Recognition 
    Using DCT coefficients as features' in ICASSP, 2015
    
    A 1D-dct is applied to the cepstral coefficients on a temporal
    sliding window.
    The resulting matrix is then flatten and reduced by using a Principal
    Component Analysis.
    
    :param cep: a matrix of cepstral cefficients, 1 line per feature vector
    :param left_ctx: number of frames to consider for left context
    :param right_ctx: number of frames to consider for right context
    :param p: a PCA matrix trained on a developpment set to reduce the
       dimension of the features. P is a portait matrix
    """
    y = numpy.r_[numpy.resize(cep[0, :], (left_ctx, cep.shape[1])),
                 cep,
                 numpy.resize(cep[-1, :], (right_ctx, cep.shape[1]))]

    ceps = framing(y, win_size=left_ctx + 1 + right_ctx).transpose(0, 2, 1)
    dct_temp = (dct_basis(left_ctx + 1 + right_ctx, left_ctx + 1 + right_ctx)).T
    if p is None:
        p = numpy.eye(dct_temp.shape[0] * cep.shape[1], dtype=numpy.float32)
    return (numpy.dot(ceps.reshape(-1, dct_temp.shape[0]),
                      dct_temp).reshape(ceps.shape[0], -1)).dot(p)


def shifted_delta_cepstral(cep, d=1, p=3, k=7):
    """
    Compute the Shifted-Delta-Cepstral features for language identification
    
    :param cep: matrix of feature, 1 vector per line
    :param d: represents the time advance and delay for the delta computation
    :param k: number of delta-cepstral blocks whose delta-cepstral 
       coefficients are stacked to form the final feature vector
    :param p: time shift between consecutive blocks.
    
    return: cepstral coefficient concatenated with shifted deltas
    """

    y = numpy.r_[numpy.resize(cep[0, :], (d, cep.shape[1])),
                 cep,
                 numpy.resize(cep[-1, :], (k * 3 + d, cep.shape[1]))]

    delta = compute_delta(y, win=d, method='diff')
    sdc = numpy.empty((cep.shape[0], cep.shape[1] * k))

    idx = numpy.zeros(delta.shape[0], dtype='bool')
    for ii in range(k):
        idx[d + ii * p] = True
    for ff in range(len(cep)):
        sdc[ff, :] = delta[idx, :].reshape(1, -1)
        idx = numpy.roll(idx, 1)
    return numpy.hstack((cep, sdc))


def trfbank(fs, nfft, lowfreq, maxfreq, nlinfilt, nlogfilt, midfreq=1000):
    """Compute triangular filterbank for cepstral coefficient computation.

    :param fs: sampling frequency of the original signal.
    :param nfft: number of points for the Fourier Transform
    :param lowfreq: lower limit of the frequency band filtered
    :param maxfreq: higher limit of the frequency band filtered
    :param nlinfilt: number of linear filters to use in low frequencies
    :param  nlogfilt: number of log-linear filters to use in high frequencies
    :param midfreq: frequency boundary between linear and log-linear filters

    :return: the filter bank and the central frequencies of each filter
    """
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    # ------------------------
    # Compute the filter bank
    # ------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    frequences = numpy.zeros(nfilt + 2, dtype=numpy.float32)
    if nlogfilt == 0:
        linsc = (maxfreq - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt + 2] = lowfreq + numpy.arange(nlinfilt + 2) * linsc
    elif nlinfilt == 0:
        low_mel = hz2mel(lowfreq)
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2)
        # mels[nlinfilt:]
        melsc = (max_mel - low_mel) / (nfilt + 1)
        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences = mel2hz(mels)
    else:
        # Compute linear filters on [0;1000Hz]
        linsc = (min([midfreq, maxfreq]) - lowfreq) / (nlinfilt + 1)
        frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
        # Compute log-linear filters on [1000;maxfreq]
        low_mel = hz2mel(min([1000, maxfreq]))
        max_mel = hz2mel(maxfreq)
        mels = numpy.zeros(nlogfilt + 2, dtype=numpy.float32)
        melsc = (max_mel - low_mel) / (nlogfilt + 1)

        # Verify that mel2hz(melsc)>linsc
        while mel2hz(melsc) < linsc:
            # in this case, we add a linear filter
            nlinfilt += 1
            nlogfilt -= 1
            frequences[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
            low_mel = hz2mel(frequences[nlinfilt - 1] + 2 * linsc)
            max_mel = hz2mel(maxfreq)
            mels = numpy.zeros(nlogfilt + 2, dtype=numpy.float32)
            melsc = (max_mel - low_mel) / (nlogfilt + 1)

        mels[:nlogfilt + 2] = low_mel + numpy.arange(nlogfilt + 2) * melsc
        # Back to the frequency domain
        frequences[nlinfilt:] = mel2hz(mels)

    heights = 2. / (frequences[2:] - frequences[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, int(numpy.floor(nfft / 2)) + 1), dtype=numpy.float32)
    # FFT bins (in Hz)
    n_frequences = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = frequences[i]
        cen = frequences[i + 1]
        hi = frequences[i + 2]

        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1, numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        left_slope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1,
                           min(numpy.floor(hi * nfft / fs) + 1, nfft), dtype=numpy.int)
        right_slope = heights[i] / (hi - cen)
        fbank[i][lid] = left_slope * (n_frequences[lid] - low)
        fbank[i][rid[:-1]] = right_slope * (hi - n_frequences[rid[:-1]])

    return fbank, frequences


def mel_filter_bank(fs, nfft, lowfreq, maxfreq, widest_nlogfilt, widest_lowfreq, widest_maxfreq,):
    """Compute triangular filterbank for cepstral coefficient computation.

    :param fs: sampling frequency of the original signal.
    :param nfft: number of points for the Fourier Transform
    :param lowfreq: lower limit of the frequency band filtered
    :param maxfreq: higher limit of the frequency band filtered
    :param widest_nlogfilt: number of log filters
    :param widest_lowfreq: lower frequency of the filter bank
    :param widest_maxfreq: higher frequency of the filter bank
    :param widest_maxfreq: higher frequency of the filter bank

    :return: the filter bank and the central frequencies of each filter
    """

    # ------------------------
    # Compute the filter bank
    # ------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    widest_freqs = numpy.zeros(widest_nlogfilt + 2, dtype=numpy.float32)
    low_mel = hz2mel(widest_lowfreq)
    max_mel = hz2mel(widest_maxfreq)
    mels = numpy.zeros(widest_nlogfilt+2)
    melsc = (max_mel - low_mel) / (widest_nlogfilt + 1)
    mels[:widest_nlogfilt + 2] = low_mel + numpy.arange(widest_nlogfilt + 2) * melsc
    # Back to the frequency domain
    widest_freqs = mel2hz(mels)

    # Select filters in the narrow band
    sub_band_freqs = numpy.array([fr for fr in widest_freqs if lowfreq <= fr <= maxfreq], dtype=numpy.float32)

    heights = 2./(sub_band_freqs[2:] - sub_band_freqs[0:-2])
    nfilt = sub_band_freqs.shape[0] - 2

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, numpy.floor(nfft/2)+1), dtype=numpy.float32)
    # FFT bins (in Hz)
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        low = sub_band_freqs[i]
        cen = sub_band_freqs[i+1]
        hi = sub_band_freqs[i+2]
        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1, numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        left_slope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1, min(numpy.floor(hi * nfft / fs) + 1,
                                                                 nfft), dtype=numpy.int)
        right_slope = heights[i] / (hi - cen)
        fbank[i][lid] = left_slope * (nfreqs[lid] - low)
        fbank[i][rid[:-1]] = right_slope * (hi - nfreqs[rid[:-1]])

    return fbank, sub_band_freqs


def power_spectrum(input_sig,
                   fs=8000,
                   win_time=0.025,
                   shift=0.01,
                   prefac=0.97):
    """
    Compute the power spectrum of the signal.
    :param input_sig:
    :param fs:
    :param win_time:
    :param shift:
    :param prefac:
    :return:
    """
    window_length = int(round(win_time * fs))
    overlap = window_length - int(shift * fs)
    framed = framing(input_sig, window_length, win_shift=window_length-overlap).copy()

    # Pre-emphasis filtering is applied after framing to be consistent with stream processing
    framed = pre_emphasis(framed, prefac)

    l = framed.shape[0]
    n_fft = 2 ** int(numpy.ceil(numpy.log2(window_length)))
    # Windowing has been changed to hanning which is supposed to have less noisy sidelobes
    # ham = numpy.hamming(window_length)
    window = numpy.hanning(window_length)

    spec = numpy.ones((l, int(n_fft / 2) + 1), dtype=numpy.float32)
    log_energy = numpy.log((framed**2).sum(axis=1) + 1e-5)
    dec = 500000
    start = 0
    stop = min(dec, l)
    while start < l:
        ahan = framed[start:stop, :] * window
        mag = numpy.fft.rfft(ahan, n_fft, axis=-1)
        spec[start:stop, :] = mag.real**2 + mag.imag**2
        start = stop
        stop = min(stop + dec, l)

    return spec, log_energy


def mfcc(input_sig,
         lowfreq=100, maxfreq=8000,
         nlinfilt=0, nlogfilt=24,
         nwin=0.025,
         fs=16000,
         nceps=13,
         shift=0.01,
         get_spec=False,
         get_mspec=False,
         prefac=0.97):
    """Compute Mel Frequency Cepstral Coefficients.

    :param input_sig: input signal from which the coefficients are computed.
            Input audio is supposed to be RAW PCM 16bits
    :param lowfreq: lower limit of the frequency band filtered. 
            Default is 100Hz.
    :param maxfreq: higher limit of the frequency band filtered.
            Default is 8000Hz.
    :param nlinfilt: number of linear filters to use in low frequencies.
            Default is 0.
    :param nlogfilt: number of log-linear filters to use in high frequencies.
            Default is 24.
    :param nwin: length of the sliding window in seconds
            Default is 0.025.
    :param fs: sampling frequency of the original signal. Default is 16000Hz.
    :param nceps: number of cepstral coefficients to extract. 
            Default is 13.
    :param shift: shift between two analyses. Default is 0.01 (10ms).
    :param get_spec: boolean, if true returns the spectrogram
    :param get_mspec:  boolean, if true returns the output of the filter banks
    :param prefac: pre-emphasis filter value

    :return: the cepstral coefficients in a ndaray as well as 
            the Log-spectrum in the mel-domain in a ndarray.

    .. note:: MFCC are computed as follows:
        
            - Pre-processing in time-domain (pre-emphasizing)
            - Compute the spectrum amplitude by windowing with a Hamming window
            - Filter the signal in the spectral domain with a triangular filter-bank, whose filters are approximatively
               linearly spaced on the mel scale, and have equal bandwith in the mel scale
            - Compute the DCT of the log-spectrom
            - Log-energy is returned as first coefficient of the feature vector.
    
    For more details, refer to [Davis80]_.
    """
    # Compute power spectrum
    spec, log_energy = power_spectrum(input_sig,
                                      fs,
                                      win_time=nwin,
                                      shift=shift,
                                      prefac=prefac)

    # Filter the spectrum through the triangle filter-bank
    n_fft = 2 ** int(numpy.ceil(numpy.log2(int(round(nwin * fs)))))
    fbank = trfbank(fs, n_fft, lowfreq, maxfreq, nlinfilt, nlogfilt)[0]

    mspec = numpy.log(numpy.dot(spec, fbank.T) + 1e-5)   # A tester avec log10 et log

    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    # The C0 term is removed as it is the constant term
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, 1:nceps + 1]
    lst = list()
    lst.append(ceps)
    lst.append(log_energy)
    if get_spec:
        lst.append(spec)
    else:
        lst.append(None)
        del spec
    if get_mspec:
        lst.append(mspec)
    else:
        lst.append(None)
        del mspec

    return lst


def fft2barkmx(n_fft, fs, nfilts=0, width=1., minfreq=0., maxfreq=8000):
    """
    Generate a matrix of weights to combine FFT bins into Bark
    bins.  n_fft defines the source FFT size at sampling rate fs.
    Optional nfilts specifies the number of output bands required
    (else one per bark), and width is the constant width of each
    band in Bark (default 1).
    While wts has n_fft columns, the second half are all zero.
    Hence, Bark spectrum is fft2barkmx(n_fft,fs) * abs(fft(xincols, n_fft));
    2004-09-05  dpwe@ee.columbia.edu  based on rastamat/audspec.m

    :param n_fft: the source FFT size at sampling rate fs
    :param fs: sampling rate
    :param nfilts: number of output bands required
    :param width: constant width of each band in Bark (default 1)
    :param minfreq:
    :param maxfreq:
    :return: a matrix of weights to combine FFT bins into Bark bins
    """
    maxfreq = min(maxfreq, fs / 2.)

    min_bark = hz2bark(minfreq)
    nyqbark = hz2bark(maxfreq) - min_bark

    if nfilts == 0:
        nfilts = numpy.ceil(nyqbark) + 1

    wts = numpy.zeros((nfilts, n_fft))

    # bark per filt
    step_barks = nyqbark / (nfilts - 1)

    # Frequency of each FFT bin in Bark
    binbarks = hz2bark(numpy.arange(n_fft / 2 + 1) * fs / n_fft)

    for i in range(nfilts):
        f_bark_mid = min_bark + i * step_barks
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        lof = (binbarks - f_bark_mid - 0.5)
        hif = (binbarks - f_bark_mid + 0.5)
        wts[i, :n_fft // 2 + 1] = 10 ** (numpy.minimum(numpy.zeros_like(hif), numpy.minimum(hif, -2.5 * lof) / width))

    return wts


def fft2melmx(n_fft,
              fs=8000,
              nfilts=0,
              width=1.,
              minfreq=0,
              maxfreq=4000,
              htkmel=False,
              constamp=False):
    """
    Generate a matrix of weights to combine FFT bins into Mel
    bins.  n_fft defines the source FFT size at sampling rate fs.
    Optional nfilts specifies the number of output bands required
    (else one per "mel/width"), and width is the constant width of each
    band relative to standard Mel (default 1).
    While wts has n_fft columns, the second half are all zero.
    Hence, Mel spectrum is fft2melmx(n_fft,fs)*abs(fft(xincols,n_fft));
    minfreq is the frequency (in Hz) of the lowest band edge;
    default is 0, but 133.33 is a common standard (to skip LF).
    maxfreq is frequency in Hz of upper edge; default fs/2.
    You can exactly duplicate the mel matrix in Slaney's mfcc.m
    as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
    htkmel=1 means use HTK's version of the mel curve, not Slaney's.
    constamp=1 means make integration windows peak at 1, not sum to 1.
    frqs returns bin center frqs.

    % 2004-09-05  dpwe@ee.columbia.edu  based on fft2barkmx

    :param n_fft:
    :param fs:
    :param nfilts:
    :param width:
    :param minfreq:
    :param maxfreq:
    :param htkmel:
    :param constamp:
    :return:
    """
    maxfreq = min(maxfreq, fs / 2.)

    if nfilts == 0:
        nfilts = numpy.ceil(hz2mel(maxfreq, htkmel) / 2.)

    wts = numpy.zeros((nfilts, n_fft))

    # Center freqs of each FFT bin
    fftfrqs = numpy.arange(n_fft / 2 + 1) / n_fft * fs

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfreq, htkmel)
    maxmel = hz2mel(maxfreq, htkmel)
    binfrqs = mel2hz(minmel +  numpy.arange(nfilts + 2) / (nfilts + 1) * (maxmel - minmel), htkmel)

    for i in range(nfilts):
        _fs = binfrqs[i + numpy.arange(3, dtype=int)]
        # scale by width
        _fs = _fs[1] + width * (_fs - _fs[1])
        # lower and upper slopes for all bins
        loslope = (fftfrqs - _fs[0]) / (_fs[1] - __fs[0])
        hislope = (_fs[2] - fftfrqs)/(_fs[2] - _fs[1])

        wts[i, 1 + numpy.arange(n_fft//2 + 1)] =numpy.maximum(numpy.zeros_like(loslope),numpy.minimum(loslope, hislope))

    if not constamp:
        # Slaney-style mel is scaled to be approx constant E per channel
        wts = numpy.dot(numpy.diag(2. / (binfrqs[2 + numpy.arange(nfilts)] - binfrqs[numpy.arange(nfilts)])) , wts)

    # Make sure 2nd half of FFT is zero
    wts[:, n_fft // 2 + 1: n_fft] = 0

    return wts, binfrqs


def audspec(power_spectrum,
            fs=16000,
            nfilts=None,
            fbtype='bark',
            minfreq=0,
            maxfreq=8000,
            sumpower=True,
            bwidth=1.):
    """

    :param power_spectrum:
    :param fs:
    :param nfilts:
    :param fbtype:
    :param minfreq:
    :param maxfreq:
    :param sumpower:
    :param bwidth:
    :return:
    """
    if nfilts is None:
        nfilts = int(numpy.ceil(hz2bark(fs / 2)) + 1)

    if not fs == 16000:
        maxfreq = min(fs / 2, maxfreq)

    nframes, nfreqs = power_spectrum.shape
    n_fft = (nfreqs -1 ) * 2

    if fbtype == 'bark':
        wts = fft2barkmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'mel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'htkmel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq, True, True)
    elif fbtype == 'fcmel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq, True, False)
    else:
        print('fbtype {} not recognized'.format(fbtype))

    wts = wts[:, :nfreqs]

    if sumpower:
        audio_spectrum = power_spectrum.dot(wts.T)
    else:
        audio_spectrum = numpy.dot(numpy.sqrt(power_spectrum), wts.T)**2

    return audio_spectrum, wts


def postaud(x, fmax, fbtype='bark', broaden=0):
    """
    do loudness equalization and cube root compression

    :param x:
    :param fmax:
    :param fbtype:
    :param broaden:
    :return:
    """
    nframes, nbands = x.shape

    # Include frequency points at extremes, discard later
    nfpts = nbands + 2 * broaden

    if fbtype == 'bark':
        bandcfhz = bark2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'mel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'htkmel' or fbtype == 'fcmel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2mel(fmax,1), num=nfpts),1)
    else:
        print('unknown fbtype {}'.format(fbtype))

    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[broaden:(nfpts - broaden)]

    # Hynek's magic equal-loudness-curve formula
    fsq = bandcfhz ** 2
    ftmp = fsq + 1.6e5
    eql = ((fsq / ftmp) ** 2) * ((fsq + 1.44e6) / (fsq + 9.61e6))

    # weight the critical bands
    z = numpy.matlib.repmat(eql.T,nframes,1) * x

    # cube root compress
    z = z ** .33

    # replicate first and last band (because they are unreliable as calculated)
    if broaden == 1:
      y = z[:, numpy.hstack((0,numpy.arange(nbands), nbands - 1))]
    else:
      y = z[:, numpy.hstack((1,numpy.arange(1, nbands - 1), nbands - 2))]

    return y, eql


def dolpc(x, model_order=8):
    """
    compute autoregressive model from spectral magnitude samples

    :param x:
    :param model_order:
    :return:
    """
    nframes, nbands = x.shape

    r = numpy.real(numpy.fft.ifft(numpy.hstack((x,x[:,numpy.arange(nbands-2,0,-1)]))))

    # First half only
    r = r[:, :nbands]

    # Find LPC coeffs by Levinson-Durbin recursion
    y_lpc = numpy.ones((r.shape[0], model_order + 1))

    for ff in range(r.shape[0]):
        y_lpc[ff, 1:], e, _ = levinson(r[ff, :-1].T, order=model_order, allow_singularity=True)
        # Normalize each poly by gain
        y_lpc[ff, :] /= e

    return y_lpc


def lpc2cep(a, nout):
    """
    Convert the LPC 'a' coefficients in each column of lpcas
    into frames of cepstra.
    nout is number of cepstra to produce, defaults to size(lpcas,1)
    2003-04-11 dpwe@ee.columbia.edu

    :param a:
    :param nout:
    :return:
    """
    ncol , nin = a.shape

    order = nin - 1

    if nout is None:
        nout = order + 1

    c = numpy.zeros((ncol, nout))

    # First cep is log(Error) from Durbin
    c[:, 0] = -numpy.log(a[:, 0])

    # Renormalize lpc A coeffs
    a /= numpy.tile(a[:, 0][:, None], (1, nin))

    for n in range(1, nout):
        sum = 0
        for m in range(1, n):
            sum += (n - m)  * a[:, m] * c[:, n - m]
        c[:, n] = -(a[:, n] + sum / n)

    return c


def lpc2spec(lpcas, nout=17):
    """
    Convert LPC coeffs back into spectra
    nout is number of freq channels, default 17 (i.e. for 8 kHz)

    :param lpcas:
    :param nout:
    :return:
    """
    [cols, rows] = lpcas.shape
    order = rows - 1

    gg = lpcas[:, 0]
    aa = lpcas / numpy.tile(gg, (rows,1)).T

    # Calculate the actual z-plane polyvals: nout points around unit circle
    zz = numpy.exp((-1j * numpy.pi / (nout - 1)) * numpy.outer(numpy.arange(nout).T,  numpy.arange(order + 1)))

    # Actual polyvals, in power (mag^2)
    features = ( 1./numpy.abs(aa.dot(zz.T))**2) / numpy.tile(gg, (nout, 1)).T

    F = numpy.zeros((cols, rows-1))
    M = numpy.zeros((cols, rows-1))

    for c in range(cols):
        aaa = aa[c, :]
        rr = numpy.roots(aaa)
        ff = numpy.angle(rr.T)
        zz = numpy.exp(1j * numpy.outer(ff, numpy.arange(len(aaa))))
        mags = numpy.sqrt(((1./numpy.abs(zz.dot(aaa)))**2)/gg[c])
        ix = numpy.argsort(ff)
        keep = ff[ix] > 0
        ix = ix[keep]
        F[c, numpy.arange(len(ix))] = ff[ix]
        M[c, numpy.arange(len(ix))] = mags[ix]

    F = F[:, F.sum(axis=0) != 0]
    M = M[:, M.sum(axis=0) != 0]

    return features, F, M


def spec2cep(spec, ncep=13, type=2):
    """
    Calculate cepstra from spectral samples (in columns of spec)
    Return ncep cepstral rows (defaults to 9)
    This one does type II dct, or type I if type is specified as 1
    dctm returns the DCT matrix that spec was multiplied by to give cep.

    :param spec:
    :param ncep:
    :param type:
    :return:
    """
    nrow, ncol = spec.shape

    # Make the DCT matrix
    dctm = numpy.zeros(ncep, nrow);
    #if type == 2 || type == 3
    #    # this is the orthogonal one, the one you want
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[1:2:(2*nrow-1)]/(2*nrow)*pi) * sqrt(2/nrow);

    #    if type == 2
    #        # make it unitary! (but not for HTK type 3)
    #        dctm(1,:) = dctm(1,:)/sqrt(2);

    #elif type == 4:  # type 1 with implicit repeating of first, last bins
    #    """
    #    Deep in the heart of the rasta/feacalc code, there is the logic
    #    that the first and last auditory bands extend beyond the edge of
    #    the actual spectra, and they are thus copied from their neighbors.
    #    Normally, we just ignore those bands and take the 19 in the middle,
    #    but when feacalc calculates mfccs, it actually takes the cepstrum
    #    over the spectrum *including* the repeated bins at each end.
    #    Here, we simulate 'repeating' the bins and an nrow+2-length
    #    spectrum by adding in extra DCT weight to the first and last
    #    bins.
    #    """
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[1:nrow]/(nrow+1)*pi) * 2;
    #        # Add in edge points at ends (includes fixup scale)
    #        dctm(i,1) = dctm(i,1) + 1;
    #        dctm(i,nrow) = dctm(i,nrow) + ((-1)^(i-1));

    #   dctm = dctm / (2*(nrow+1));
    #else % dpwe type 1 - same as old spec2cep that expanded & used fft
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[0:(nrow-1)]/(nrow-1)*pi) * 2 / (2*(nrow-1));
    #    dctm(:,[1 nrow]) = dctm(:, [1 nrow])/2;

    #cep = dctm*log(spec);
    return None, None, None


def lifter(x, lift=0.6, invs=False):
    """
    Apply lifter to matrix of cepstra (one per column)
    lift = exponent of x i^n liftering
    or, as a negative integer, the length of HTK-style sin-curve liftering.
    If inverse == 1 (default 0), undo the liftering.

    :param x:
    :param lift:
    :param invs:
    :return:
    """
    nfrm , ncep = x.shape

    if lift == 0:
        y = x
    else:
        if lift > 0:
            if lift > 10:
                print('Unlikely lift exponent of {} did you mean -ve?'.format(lift))
            liftwts = numpy.hstack((1, numpy.arange(1, ncep)**lift))

        elif lift < 0:
        # Hack to support HTK liftering
            L = float(-lift)
            if (L != numpy.round(L)):
                print('HTK liftering value {} must be integer'.format(L))

            liftwts = numpy.hstack((1, 1 + L/2*numpy.sin(numpy.arange(1, ncep) * numpy.pi / L)))

        if invs:
            liftwts = 1 / liftwts

        y = x.dot(numpy.diag(liftwts))

    return y


def plp(input_sig,
         nwin=0.025,
         fs=16000,
         plp_order=13,
         shift=0.01,
         get_spec=False,
         get_mspec=False,
         prefac=0.97,
         rasta=True):
    """
    output is matrix of features, row = feature, col = frame

    % fs is sampling rate of samples, defaults to 8000
    % dorasta defaults to 1; if 0, just calculate PLP
    % modelorder is order of PLP model, defaults to 8.  0 -> no PLP

    :param input_sig:
    :param fs: sampling rate of samples default is 8000
    :param rasta: default is True, if False, juste compute PLP
    :param model_order: order of the PLP model, default is 8, 0 means no PLP

    :return: matrix of features, row = features, column are frames
    """
    plp_order -= 1

    # first compute power spectrum
    powspec, log_energy = power_spectrum(input_sig, fs, nwin, shift, prefac)

    # next group to critical bands
    audio_spectrum = audspec(powspec, fs)[0]
    nbands = audio_spectrum.shape[0]

    if rasta:
        # put in log domain
        nl_aspectrum = numpy.log(audio_spectrum)

        #  next do rasta filtering
        ras_nl_aspectrum = rasta_filt(nl_aspectrum)

        # do inverse log
        audio_spectrum = numpy.exp(ras_nl_aspectrum)

    # do final auditory compressions
    post_spectrum = postaud(audio_spectrum, fs / 2.)[0]

    if plp_order > 0:

        # LPC analysis
        lpcas = dolpc(post_spectrum, plp_order)

        # convert lpc to cepstra
        cepstra = lpc2cep(lpcas, plp_order + 1)

        # .. or to spectra
        spectra, F, M = lpc2spec(lpcas, nbands)

    else:

        # No LPC smoothing of spectrum
        spectra = post_spectrum
        cepstra = spec2cep(spectra)

    cepstra = lifter(cepstra, 0.6)

    lst = list()
    lst.append(cepstra)
    lst.append(log_energy)
    if get_spec:
        lst.append(powspec)
    else:
        lst.append(None)
        del powspec
    if get_mspec:
        lst.append(post_spectrum)
    else:
        lst.append(None)
        del post_spectrum

    return lst


def framing(sig, win_size, win_shift=1, context=(0, 0), pad='zeros'):
    """
    :param sig: input signal, can be mono or multi dimensional
    :param win_size: size of the window in term of samples
    :param win_shift: shift of the sliding window in terme of samples
    :param context: tuple of left and right context
    :param pad: can be zeros or edge
    """
    dsize = sig.dtype.itemsize
    if sig.ndim == 1:
        sig = sig[:, numpy.newaxis]
    # Manage padding
    c = (context, ) + (sig.ndim - 1) * ((0, 0), )
    _win_size = win_size + sum(context)
    shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
    strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    if pad == 'zeros':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                                  shape=shape,
                                                  strides=strides).squeeze()
    elif pad == 'edge':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'edge'),
                                                  shape=shape,
                                                  strides=strides).squeeze()


def dct_basis(nbasis, length):
    """
    :param nbasis: number of CT coefficients to keep
    :param length: length of the matrix to process
    :return: a basis of DCT coefficients
    """
    return scipy.fftpack.idct(numpy.eye(nbasis, length), norm='ortho')


def levinson(r, order=None, allow_singularity=False):
    r"""Levinson-Durbin recursion.

    Find the coefficients of a length(r)-1 order autoregressive linear process

    :param r: autocorrelation sequence of length N + 1 (first element being the zero-lag autocorrelation)
    :param order: requested order of the autoregressive coefficients. default is N.
    :param allow_singularity: false by default. Other implementations may be True (e.g., octave)

    :return:
        * the `N+1` autoregressive coefficients :math:`A=(1, a_1...a_N)`
        * the prediction errors
        * the `N` reflections coefficients values

    This algorithm solves the set of complex linear simultaneous equations
    using Levinson algorithm.

    .. math::

        \bold{T}_M \left( \begin{array}{c} 1 \\ \bold{a}_M \end{array} \right) =
        \left( \begin{array}{c} \rho_M \\ \bold{0}_M  \end{array} \right)

    where :math:`\bold{T}_M` is a Hermitian Toeplitz matrix with elements
    :math:`T_0, T_1, \dots ,T_M`.

    .. note:: Solving this equations by Gaussian elimination would
        require :math:`M^3` operations whereas the levinson algorithm
        requires :math:`M^2+M` additions and :math:`M^2+M` multiplications.

    This is equivalent to solve the following symmetric Toeplitz system of
    linear equations

    .. math::

        \left( \begin{array}{cccc}
        r_1 & r_2^* & \dots & r_{n}^*\\
        r_2 & r_1^* & \dots & r_{n-1}^*\\
        \dots & \dots & \dots & \dots\\
        r_n & \dots & r_2 & r_1 \end{array} \right)
        \left( \begin{array}{cccc}
        a_2\\
        a_3 \\
        \dots \\
        a_{N+1}  \end{array} \right)
        =
        \left( \begin{array}{cccc}
        -r_2\\
        -r_3 \\
        \dots \\
        -r_{N+1}  \end{array} \right)

    where :math:`r = (r_1  ... r_{N+1})` is the input autocorrelation vector, and
    :math:`r_i^*` denotes the complex conjugate of :math:`r_i`. The input r is typically
    a vector of autocorrelation coefficients where lag 0 is the first
    element :math:`r_1`.


    .. doctest::

        >>> import numpy; from spectrum import LEVINSON
        >>> T = numpy.array([3., -2+0.5j, .7-1j])
        >>> a, e, k = LEVINSON(T)

    """
    #from numpy import isrealobj
    T0  = numpy.real(r[0])
    T = r[1:]
    M = len(T)

    if order is None:
        M = len(T)
    else:
        assert order <= M, 'order must be less than size of the input data'
        M = order

    realdata = numpy.isrealobj(r)
    if realdata is True:
        A = numpy.zeros(M, dtype=float)
        ref = numpy.zeros(M, dtype=float)
    else:
        A = numpy.zeros(M, dtype=complex)
        ref = numpy.zeros(M, dtype=complex)

    P = T0

    for k in range(M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            #save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + A[j] * T[k-j-1]
            temp = -save / P
        if realdata:
            P = P * (1. - temp**2.)
        else:
            P = P * (1. - (temp.real**2+temp.imag**2))

        if (P <= 0).any() and allow_singularity==False:
            raise ValueError("singular matrix")
        A[k] = temp
        ref[k] = temp # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k+1)//2
        if realdata is True:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj]
                if j != kj:
                    A[kj] += temp*save
        else:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj].conjugate()
                if j != kj:
                    A[kj] = A[kj] + temp * save.conjugate()

    return A, P, ref


def sum_log_probabilities(lp):
    """Sum log probabilities in a secure manner to avoid extreme values

    :param lp: numpy array of log-probabilities to sum
    """
    pp_max = numpy.max(lp, axis=1)
    log_lk = pp_max + numpy.log(numpy.sum(numpy.exp((lp.transpose() - pp_max).T), axis=1))
    ind = ~numpy.isfinite(pp_max)
    if sum(ind) != 0:
        log_lk[ind] = pp_max[ind]
    pp = numpy.exp((lp.transpose() - log_lk).transpose())
    llk = log_lk.sum()
    return pp, llk


class Mixture(object):
    """
    A class for Gaussian Mixture Model storage.
    For more details about Gaussian Mixture Models (GMM) you can refer to
    [Bimbot04]_.
    
    :attr w: array of weight parameters
    :attr mu: ndarray of mean parameters, each line is one distribution 
    :attr invcov: ndarray of inverse co-variance parameters, 2-dimensional 
        for diagonal co-variance distribution 3-dimensional for full co-variance
    :attr invchol: 3-dimensional ndarray containing upper cholesky
        decomposition of the inverse co-variance matrices
    :attr cst: array of constant computed for each distribution
    :attr det: array of determinant for each distribution
    
    """
    @staticmethod
    def read_alize(file_name):
        """

        :param file_name:
        :return:
        """
        """Read a Mixture in alize raw format

        :param mixtureFileName: name of the file to read from
        """
        mixture = Mixture()

        with open(file_name, 'rb') as f:
            distrib_nb = struct.unpack("I", f.read(4))[0]
            vect_size = struct.unpack("<I", f.read(4))[0]

            # resize all attributes
            mixture.w = numpy.zeros(distrib_nb, "d")
            mixture.invcov = numpy.zeros((distrib_nb, vect_size), "d")
            mixture.mu = numpy.zeros((distrib_nb, vect_size), "d")
            mixture.cst = numpy.zeros(distrib_nb, "d")
            mixture.det = numpy.zeros(distrib_nb, "d")

            for d in range(distrib_nb):
                mixture.w[d] = struct.unpack("<d", f.read(8))[0]
            for d in range(distrib_nb):
                mixture.cst[d] = struct.unpack("d", f.read(8))[0]
                mixture.det[d] = struct.unpack("d", f.read(8))[0]
                f.read(1)
                for c in range(vect_size):
                    mixture.invcov[d, c] = struct.unpack("d", f.read(8))[0]
                for c in range(vect_size):
                    mixture.mu[d, c] = struct.unpack("d", f.read(8))[0]
        mixture._compute_all()
        return mixture

    @staticmethod
    def read_htk(filename, begin_hmm=False, state2=False):
        """Read a Mixture in HTK format

        :param filename: name of the file to read from
        :param begin_hmm: boolean
        :param state2: boolean
        """
        mixture = Mixture()
        with open(filename, 'rb') as f:
            lines = [line.rstrip() for line in f]

        distrib = 0
        vect_size = 0
        for i in range(len(lines)):

            if lines[i] == '':
                break

            w = lines[i].split()

            if w[0] == '<NUMMIXES>':
                distrib_nb = int(w[1])
                mixture.w.resize(distrib_nb)
                mixture.cst.resize(distrib_nb)
                mixture.det.resize(distrib_nb)

            if w[0] == '<BEGINHMM>':
                begin_hmm = True

            if w[0] == '<STATE>':
                state2 = True

            if begin_hmm & state2:

                if w[0].upper() == '<MIXTURE>':
                    distrib = int(w[1]) - 1
                    mixture.w[distrib] = numpy.double(w[2])

                elif w[0].upper() == '<MEAN>':
                    if vect_size == 0:
                        vect_size = int(w[1])
                    mixture.mu.resize(distrib_nb, vect_size)
                    i += 1
                    mixture.mu[distrib, :] = numpy.double(lines[i].split())

                elif w[0].upper() == '<VARIANCE>':
                    if mixture.invcov.shape[0] == 0:
                        vect_size = int(w[1])
                    mixture.invcov.resize(distrib_nb, vect_size)
                    i += 1
                    C = numpy.double(lines[i].split())
                    mixture.invcov[distrib, :] = 1 / C

                elif w[0].upper() == '<INVCOVAR>':
                    raise Exception("we don't manage full covariance model")
                elif w[0].upper() == '<GCONST>':
                    mixture.cst[distrib] = numpy.exp(-.05 * numpy.double(w[1]))
        mixture._compute_all()
        return mixture

    def __init__(self,
                 mixture_file_name='',
                 name='empty'):
        """Initialize a Mixture from a file or as an empty Mixture.
        
        :param mixture_file_name: name of the file to read from, if empty, initialize
            an empty mixture
        """
        self.w = numpy.array([])
        self.mu = numpy.array([])
        self.invcov = numpy.array([])
        self.invchol = numpy.array([])
        self.cov_var_ctl = numpy.array([])
        self.cst = numpy.array([])
        self.det = numpy.array([])
        self.name = name
        self.A = 0

        if mixture_file_name != '':
            self.read(mixture_file_name)

    def __add__(self, other):
        """Overide the sum for a mixture.
        Weight, means and inv_covariances are added, det and cst are
        set to 0
        """
        new_mixture = Mixture()
        new_mixture.w = self.w + other.w
        new_mixture.mu = self.mu + other.mu
        new_mixture.invcov = self.invcov + other.invcov
        return new_mixture

    def init_from_diag(self, diag_mixture):
        """

        :param diag_mixture:
        """
        distrib_nb = diag_mixture.w.shape[0]
        dim = diag_mixture.mu.shape[1]

        self.w = diag_mixture.w
        self.cst = diag_mixture.cst
        self.det = diag_mixture.det
        self.mu = diag_mixture.mu

        self.invcov = numpy.empty((distrib_nb, dim, dim))
        self.invchol = numpy.empty((distrib_nb, dim, dim))
        for gg in range(distrib_nb):
            self.invcov[gg] = numpy.diag(diag_mixture.invcov[gg, :])
            self.invchol[gg] = numpy.linalg.cholesky(self.invcov[gg])
            self.cov_var_ctl = numpy.diag(diag_mixture.cov_var_ctl)
        self.name = diag_mixture.name
        self.A = numpy.zeros(self.cst.shape)  # we keep zero here as it is not used for full covariance distributions

    def _serialize(self):
        """
        Serialization is necessary to share the memomry when running multiprocesses
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            sh = self.w.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.w.size)
            self.w = numpy.ctypeslib.as_array(tmp.get_obj())
            self.w = self.w.reshape(sh)

            sh = self.mu.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.mu.size)
            self.mu = numpy.ctypeslib.as_array(tmp.get_obj())
            self.mu = self.mu.reshape(sh)

            sh = self.invcov.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.invcov.size)
            self.invcov = numpy.ctypeslib.as_array(tmp.get_obj())
            self.invcov = self.invcov.reshape(sh)

            sh = self.cov_var_ctl.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.cov_var_ctl.size)
            self.cov_var_ctl = numpy.ctypeslib.as_array(tmp.get_obj())
            self.cov_var_ctl = self.cov_var_ctl.reshape(sh)

            sh = self.cst.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.cst.size)
            self.cst = numpy.ctypeslib.as_array(tmp.get_obj())
            self.cst = self.cst.reshape(sh)

            sh = self.det.shape
            tmp = multiprocessing.Array(ctypes.c_double, self.det.size)
            self.det = numpy.ctypeslib.as_array(tmp.get_obj())
            self.det = self.det.reshape(sh)

    def get_distrib_nb(self):
        """
        Return the number of Gaussian distributions in the mixture
        :return: then number of distributions
        """
        return self.w.shape[0]

    def read(self, mixture_file_name, prefix=''):
        """Read a Mixture in hdf5 format

        :param mixture_file_name: name of the file to read from
        :param prefix:
        """
        with h5py.File(mixture_file_name, 'r') as f:
            self.w = f.get(prefix+'w').value
            self.w.resize(numpy.max(self.w.shape))
            self.mu = f.get(prefix+'mu').value
            self.invcov = f.get(prefix+'invcov').value
            self.invchol = f.get(prefix+'invchol').value
            self.cov_var_ctl = f.get(prefix+'cov_var_ctl').value
            self.cst = f.get(prefix+'cst').value
            self.det = f.get(prefix+'det').value
            self.A = f.get(prefix+'a').value

    def write_alize(self, mixture_file_name):
        """Save a mixture in alize raw format

        :param mixture_file_name: name of the file to write in
        """
        with open(mixture_file_name, 'wb') as of:
            # write the number of distributions per state
            of.write(struct.pack("<I", self.distrib_nb()))
            # Write the dimension of the features
            of.write(struct.pack("<I", self.dim()))
            # Weights
            of.write(struct.pack("<" + "d" * self.w.shape[0], *self.w))
            # For each distribution
            for d in range(self.distrib_nb()):
                # Write the constant
                of.write(struct.pack("<d", self.cst[d]))
                # Write the determinant
                of.write(struct.pack("<d", self.det[d]))
                # write a meaningless char for compatibility purpose
                of.write(struct.pack("<c", bytes(1)))
                # Covariance
                of.write(
                    struct.pack("<" + "d" * self.dim(), *self.invcov[d, :]))
                # Means
                of.write(struct.pack("<" + "d" * self.dim(), *self.mu[d, :]))

    def write(self, mixture_file_name, prefix='', mode='w'):
        """Save a Mixture in hdf5 format

        :param mixture_file_name: the name of the file to write in
        :param prefix: prefix of the group in the HDF5 file
        :param mode: mode of the opening, default is "w"
        """
        f = h5py.File(mixture_file_name, mode)

        f.create_dataset(prefix+'w', self.w.shape, "d", self.w,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'mu', self.mu.shape, "d", self.mu,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'invcov', self.invcov.shape, "d", self.invcov,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'invchol', self.invchol.shape, "d", self.invchol,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'cov_var_ctl', self.cov_var_ctl.shape, "d",
                         self.cov_var_ctl,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'cst', self.cst.shape, "d", self.cst,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'det', self.det.shape, "d", self.det,
                         compression="gzip",
                         fletcher32=True)
        f.create_dataset(prefix+'a', self.A.shape, "d", self.A,
                         compression="gzip",
                         fletcher32=True)
        f.close()

    def distrib_nb(self):
        """Return the number of distribution of the Mixture
        
        :return: the number of distribution in the Mixture
        """
        return self.w.shape[0]

    def dim(self):
        """Return the dimension of distributions of the Mixture
        
        :return: an integer, size of the acoustic vectors
        """
        return self.mu.shape[1]

    def sv_size(self):
        """Return the dimension of the super-vector
        
        :return: an integer, size of the mean super-vector
        """
        return self.mu.shape[1] * self.w.shape[0]

    def _compute_all(self):
        """Compute determinant and constant values for each distribution"""
        if self.invcov.ndim == 2:  # for Diagonal covariance only
            self.det = 1.0 / numpy.prod(self.invcov, axis=1)
        elif self.invcov.ndim == 3:  # For full covariance dstributions
            for gg in range(self.mu.shape[0]):
                self.det[gg] = 1./numpy.linalg.det(self.invcov[gg])
                self.invchol[gg] = numpy.linalg.cholesky(self.invcov[gg])

        self.cst = 1.0 / (numpy.sqrt(self.det) * (2.0 * numpy.pi) ** (self.dim() / 2.0))

        if self.invcov.ndim == 2:
            self.A = (numpy.square(self.mu) * self.invcov).sum(1) - 2.0 * (numpy.log(self.w) + numpy.log(self.cst))
        elif self.invcov.ndim == 3:
            self.A = numpy.zeros(self.cst.shape)

    def validate(self):
        """Verify the format of the Mixture
        
        :return: a boolean giving the status of the Mixture
        """
        cov = 'diag'
        ok = (self.w.ndim == 1)
        ok &= (self.det.ndim == 1)
        ok &= (self.cst.ndim == 1)
        ok &= (self.mu.ndim == 2)
        if self.invcov.ndim == 3:
            cov = 'full'
        else:
            ok &= (self.invcov.ndim == 2)

        ok &= (self.w.shape[0] == self.mu.shape[0])
        ok &= (self.w.shape[0] == self.cst.shape[0])
        ok &= (self.w.shape[0] == self.det.shape[0])
        if cov == 'diag':
            ok &= (self.invcov.shape == self.mu.shape)
        else:
            ok &= (self.w.shape[0] == self.invcov.shape[0])
            ok &= (self.mu.shape[1] == self.invcov.shape[1])
            ok &= (self.mu.shape[1] == self.invcov.shape[2])
        return ok

    def get_mean_super_vector(self):
        """Return mean super-vector
        
        :return: an array, super-vector of the mean coefficients
        """
        sv = self.mu.flatten()
        return sv

    def get_invcov_super_vector(self):
        """Return Inverse covariance super-vector
        
        :return: an array, super-vector of the inverse co-variance coefficients
        """
        assert self.invcov.ndim == 2, 'Must be diagonal co-variance.'
        sv = self.invcov.flatten()
        return sv

    def compute_log_posterior_probabilities_full(self, cep, mu=None):
        """ Compute log posterior probabilities for a set of feature frames.

        :param cep: a set of feature frames in a ndarray, one feature per row
        :param mu: a mean super-vector to replace the ubm's one. If it is an empty
              vector, use the UBM

        :return: A ndarray of log-posterior probabilities corresponding to the
              input feature set.
        """
        if cep.ndim == 1:
            cep = cep[:, numpy.newaxis]
        if mu is None:
            mu = self.mu
        tmp = (cep - mu[:, numpy.newaxis, :])
        a = numpy.einsum('ijk,imk->ijm', tmp, self.invchol)
        lp = numpy.log(self.w[:, numpy.newaxis]) + numpy.log(self.cst[:, numpy.newaxis]) - 0.5 * (a * a).sum(-1)

        return lp.T

    def compute_log_posterior_probabilities(self, cep, mu=None):
        """ Compute log posterior probabilities for a set of feature frames.
        
        :param cep: a set of feature frames in a ndarray, one feature per row
        :param mu: a mean super-vector to replace the ubm's one. If it is an empty 
              vector, use the UBM
        
        :return: A ndarray of log-posterior probabilities corresponding to the 
              input feature set.
        """
        if cep.ndim == 1:
            cep = cep[numpy.newaxis, :]
        A = self.A
        if mu is None:
            mu = self.mu
        else:
            # for MAP, Compute the data independent term
            A = (numpy.square(mu.reshape(self.mu.shape)) * self.invcov).sum(1) \
               - 2.0 * (numpy.log(self.w) + numpy.log(self.cst))

        # Compute the data independent term
        B = numpy.dot(numpy.square(cep), self.invcov.T) \
            - 2.0 * numpy.dot(cep, numpy.transpose(mu.reshape(self.mu.shape) * self.invcov))

        # Compute the exponential term
        lp = -0.5 * (B + A)
        return lp

    @staticmethod
    def variance_control(cov, flooring, ceiling, cov_ctl):
        """variance_control for Mixture (florring and ceiling)

        :param cov: covariance to control
        :param flooring: float, florring value
        :param ceiling: float, ceiling value
        :param cov_ctl: co-variance to consider for flooring and ceiling
        """
        floor = flooring * cov_ctl
        ceil = ceiling * cov_ctl

        to_floor = numpy.less_equal(cov, floor)
        to_ceil = numpy.greater_equal(cov, ceil)

        cov[to_floor] = floor[to_floor]
        cov[to_ceil] = ceil[to_ceil]
        return cov

    def _reset(self):
        """Set all the Mixture values to ZERO"""
        self.cst.fill(0.0)
        self.det.fill(0.0)
        self.w.fill(0.0)
        self.mu.fill(0.0)
        self.invcov.fill(0.0)
        self.A = 0.0

    def _split_ditribution(self):
        """Split each distribution into two depending on the principal
            axis of variance."""
        sigma = 1.0 / self.invcov
        sig_max = numpy.max(sigma, axis=1)
        arg_max = numpy.argmax(sigma, axis=1)

        shift = numpy.zeros(self.mu.shape)
        for x, y, z in zip(range(arg_max.shape[0]), arg_max, sig_max):
            shift[x, y] = numpy.sqrt(z)

        self.mu = numpy.vstack((self.mu - shift, self.mu + shift))
        self.invcov = numpy.vstack((self.invcov, self.invcov))
        self.w = numpy.concatenate([self.w, self.w]) * 0.5
        self.cst = numpy.zeros(self.w.shape)
        self.det = numpy.zeros(self.w.shape)
        self.cov_var_ctl = numpy.vstack((self.cov_var_ctl, self.cov_var_ctl))

        self._compute_all()

    def _expectation(self, accum, cep):
        """Expectation step of the EM algorithm. Calculate the expected value 
            of the log likelihood function, with respect to the conditional 
            distribution.
        
        :param accum: a Mixture object to store the accumulated statistics
        :param cep: a set of input feature frames
        
        :return loglk: float, the log-likelihood computed over the input set of 
              feature frames.
        """
        if cep.ndim == 1:
            cep = cep[:, numpy.newaxis]
        if self.invcov.ndim == 2:
            lp = self.compute_log_posterior_probabilities(cep)
        elif self.invcov.ndim == 3:
            lp = self.compute_log_posterior_probabilities_full(cep)
        pp, loglk = sum_log_probabilities(lp)

        # zero order statistics
        accum.w += pp.sum(0)
        # first order statistics
        accum.mu += numpy.dot(cep.T, pp).T
        # second order statistics
        if self.invcov.ndim == 2:
            accum.invcov += numpy.dot(numpy.square(cep.T), pp).T  # version for diagonal covariance
        elif self.invcov.ndim == 3:
            tmp = numpy.einsum('ijk,ilk->ijl', cep[:, :, numpy.newaxis], cep[:, :, numpy.newaxis])
            accum.invcov += numpy.einsum('ijk,im->mjk', tmp, pp)

        # return the log-likelihood
        return loglk

    def _expectation_list(self, stat_acc, feature_list, feature_server, llk_acc=numpy.zeros(1), num_thread=1):
        """
        Expectation step of the EM algorithm. Calculate the expected value
        of the log likelihood function, with respect to the conditional
        distribution.

        :param stat_acc:
        :param feature_list:
        :param feature_server:
        :param llk_acc:
        :param num_thread:
        :return:
        """
        stat_acc._reset()
        feature_server.keep_all_features = False
        for feat in feature_list:
            cep = feature_server.load(feat)[0]
            llk_acc[0] += self._expectation(stat_acc, cep)

    def _maximization(self, accum, ceil_cov=10, floor_cov=1e-2):
        """Re-estimate the parmeters of the model which maximize the likelihood
            on the data.
        
        :param accum: a Mixture in which statistics computed during the E step 
              are stored
        :param floor_cov: a constant; minimum bound to consider, default is 1e-200
        """
        self.w = accum.w / numpy.sum(accum.w)
        self.mu = accum.mu / accum.w[:, numpy.newaxis]
        if accum.invcov.ndim == 2:
            cov = accum.invcov / accum.w[:, numpy.newaxis] - numpy.square(self.mu)
            cov = Mixture.variance_control(cov, floor_cov, ceil_cov, self.cov_var_ctl)
            self.invcov = 1.0 / cov
        elif accum.invcov.ndim == 3:
            cov = accum.invcov / accum.w[:, numpy.newaxis, numpy.newaxis] \
                  - numpy.einsum('ijk,ilk->ijl', self.mu[:, :, numpy.newaxis], self.mu[:, :, numpy.newaxis])
            # ADD VARIANCE CONTROL
            for gg in range(self.w.shape[0]):
                self.invcov[gg] = numpy.linalg.inv(cov[gg])
                self.invchol[gg] = numpy.linalg.cholesky(self.invcov[gg]).T
        self._compute_all()

    def _init(self, features_server, feature_list, num_thread=1):
        """
        Initialize a Mixture as a single Gaussian distribution which
        mean and covariance are computed on a set of feature frames

        :param features_server:
        :param feature_list:
        :param num_thread:
        :return:
        """

        # Init using all data
        features = features_server.stack_features_parallel(feature_list, num_thread=num_thread)
        n_frames = features.shape[0]
        mu = features.mean(0)
        cov = (features**2).mean(0)

        #n_frames, mu, cov = mean_std_many(features_server, feature_list, in_context=False, num_thread=num_thread)
        self.mu = mu[None]
        self.invcov = 1./cov[None]
        self.w = numpy.asarray([1.0])
        self.cst = numpy.zeros(self.w.shape)
        self.det = numpy.zeros(self.w.shape)
        self.cov_var_ctl = 1.0 / copy.deepcopy(self.invcov)
        self._compute_all()

    def EM_split(self,
                 features_server,
                 feature_list,
                 distrib_nb,
                 iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
                 num_thread=1,
                 llk_gain=0.01,
                 save_partial=False,
                 output_file_name="ubm",
                 ceil_cov=10,
                 floor_cov=1e-2):
        """Expectation-Maximization estimation of the Mixture parameters.
        
        :param features_server: sidekit.FeaturesServer used to load data
        :param feature_list: list of feature files to train the GMM
        :param distrib_nb: final number of distributions
        :param iterations: list of iteration number for each step of the learning process
        :param num_thread: number of thread to launch for parallel computing
        :param llk_gain: limit of the training gain. Stop the training when gain between
                two iterations is less than this value
        :param save_partial: name of the file to save intermediate mixtures,
               if True, save before each split of the distributions
        :param ceil_cov:
        :param floor_cov:
        
        :return llk: a list of log-likelihoods obtained after each iteration
        """
        llk = []

        self._init(features_server, feature_list, num_thread)

        # for N iterations:
        for it in iterations[:int(numpy.log2(distrib_nb))]:
            # Save current model before spliting
            if save_partial:
                self.write('{}_{}g.h5'.format(output_file_name, self.get_distrib_nb()), prefix='')

            self._split_ditribution()

            # initialize the accumulator
            accum = copy.deepcopy(self)

            for i in range(it):
                accum._reset()

                # serialize the accum
                accum._serialize()
                llk_acc = numpy.zeros(1)
                sh = llk_acc.shape
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                    llk_acc = numpy.ctypeslib.as_array(tmp.get_obj())
                    llk_acc = llk_acc.reshape(sh)

                logging.debug('Expectation')
                # E step
                self._expectation_list(stat_acc=accum,
                                       feature_list=feature_list,
                                       feature_server=features_server,
                                       llk_acc=llk_acc,
                                       num_thread=num_thread)
                llk.append(llk_acc[0] / numpy.sum(accum.w))

                # M step
                logging.debug('Maximisation')
                self._maximization(accum, ceil_cov=ceil_cov, floor_cov=floor_cov)
                if i > 0:
                    # gain = llk[-1] - llk[-2]
                    # if gain < llk_gain:
                        # logging.debug(
                        #    'EM (break) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        #    self.mu.shape[0], i + 1, it, gain, self.name,
                        #    len(cep))
                    #    break
                    # else:
                        # logging.debug(
                        #    'EM (continu) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        #    self.mu.shape[0], i + 1, it, gain, self.name,
                        #    len(cep))
                    #    break
                    pass
                else:
                    # logging.debug(
                    #    'EM (start) distrib_nb: %d %i/%i llk: %f -- %s, %d',
                    #    self.mu.shape[0], i + 1, it, llk[-1],
                    #    self.name, len(cep))
                    pass

        return llk

    def EM_uniform(self, cep, distrib_nb, iteration_min=3, iteration_max=10,
                   llk_gain=0.01, do_init=True):

        """Expectation-Maximization estimation of the Mixture parameters.

        :param cep: set of feature frames to consider
        :param cep: set of feature frames to consider
        :param distrib_nb: number of distributions
        :param iteration_min: minimum number of iterations to perform
        :param iteration_max: maximum number of iterations to perform
        :param llk_gain: gain in term of likelihood, stop the training when the gain is less than this value
        :param do_init: boolean, if True initialize the GMM from the training data

        :return llk: a list of log-likelihoods obtained after each iteration

        """
        llk = []

        if do_init:
            self._init_uniform(cep, distrib_nb)
        accum = copy.deepcopy(self)

        for i in range(0, iteration_max):
            accum._reset()
            # serialize the accum
            accum._serialize()
            llk_acc = numpy.zeros(1)
            sh = llk_acc.shape
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                llk_acc = numpy.ctypeslib.as_array(tmp.get_obj())
                llk_acc = llk_acc.reshape(sh)

            # E step
            # llk.append(self._expectation_parallel(accum, cep, num_thread) / cep.shape[0])
            # self._expectation(accum,cep)
            llk.append(self._expectation(accum, cep) / cep.shape[0])

            # M step
            self._maximization(accum)
            if i > 0:
                gain = llk[-1] - llk[-2]
                if gain < llk_gain and i >= iteration_min:
                    logging.debug(
                        'EM (break) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        self.mu.shape[0], i + 1, iteration_max, gain, self.name,
                        len(cep))
                    break
                else:
                    logging.debug(
                        'EM (continu) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                        self.mu.shape[0], i + 1, iteration_max, gain, self.name,
                        len(cep))
            else:
                logging.debug(
                    'EM (start) distrib_nb: %d %i/%i llk: %f -- %s, %d',
                    self.mu.shape[0], i + 1, iteration_max, llk[-1],
                    self.name, len(cep))
        return llk

    def _init_uniform(self, cep, distrib_nb):
        """

        :param cep: matrix of acoustic frames
        :param distrib_nb: number of distributions
        """

        # Load data to initialize the mixture
        # self._init(fs, cep)
        cov_tmp = copy.deepcopy(self.invcov)
        nb = cep.shape[0]
        self.w = numpy.full(distrib_nb, 1.0 / distrib_nb, "d")
        self.cst = numpy.zeros(distrib_nb, "d")
        self.det = numpy.zeros(distrib_nb, "d")

        for i in range(0, distrib_nb):
            start = nb // distrib_nb * i
            end = max(start + 10, nb)
            mean = numpy.mean(cep[start:end, :], axis=0)
            cov = (cep[start:end, :]**2).mean(0)
            if i == 0:
                self.mu = mean
                self.invcov = 1./cov[None]
            else:
                self.mu = numpy.vstack((self.mu, mean))
                self.invcov = numpy.vstack((self.invcov, cov))
        self.cov_var_ctl = 1.0 / copy.deepcopy(self.invcov)

        self._compute_all()

    def EM_diag2full(self, diagonal_mixture, features_server, featureList, iterations=2, num_thread=1):
        """Expectation-Maximization estimation of the Mixture parameters.

        :param features_server: sidekit.FeaturesServer used to load data
        :param featureList: list of feature files to train the GMM
        :param iterations: list of iteration number for each step of the learning process
        :param num_thread: number of thread to launch for parallel computing

        :return llk: a list of log-likelihoods obtained after each iteration
        """
        llk = []

        # Convert the covariance matrices into full ones
        distrib_nb = diagonal_mixture.w.shape[0]
        dim = diagonal_mixture.mu.shape[1]

        self.w = diagonal_mixture.w
        self.cst = diagonal_mixture.cst
        self.det = diagonal_mixture.det
        self.mu = diagonal_mixture.mu

        self.invcov = numpy.empty((distrib_nb, dim, dim))
        self.invchol = numpy.empty((distrib_nb, dim, dim))
        for gg in range(distrib_nb):
            self.invcov[gg] = numpy.diag(diagonal_mixture.invcov[gg, :])
            self.invchol[gg] = numpy.linalg.cholesky(self.invcov[gg])
            self.cov_var_ctl = numpy.diag(diagonal_mixture.cov_var_ctl)
        self.name = diagonal_mixture.name
        self.A = numpy.zeros(self.cst.shape)  # we keep zero here as it is not used for full covariance distributions

        # Create Accumulator
        accum = copy.deepcopy(self)

        # Run iterations of EM
        for it in range(iterations):
            logging.debug('EM convert full it: %d', it)

            accum._reset()

            # serialize the accum
            accum._serialize()
            llk_acc = numpy.zeros(1)
            sh = llk_acc.shape
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                tmp = multiprocessing.Array(ctypes.c_double, llk_acc.size)
                llk_acc = numpy.ctypeslib.as_array(tmp.get_obj())
                llk_acc = llk_acc.reshape(sh)

            logging.debug('Expectation')
            # E step
            self._expectation_list(stat_acc=accum,
                                   feature_list=featureList,
                                   feature_server=features_server,
                                   llk_acc=llk_acc,
                                   num_thread=num_thread)
            llk.append(llk_acc[0] / numpy.sum(accum.w))

            # M step
            logging.debug('Maximisation')
            self._maximization(accum)
            if it > 0:
                # gain = llk[-1] - llk[-2]
                # if gain < llk_gain:
                    # logging.debug(
                    #    'EM (break) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                    #    self.mu.shape[0], i + 1, it, gain, self.name,
                    #    len(cep))
                #    break
                # else:
                    # logging.debug(
                    #    'EM (continu) distrib_nb: %d %i/%d gain: %f -- %s, %d',
                    #    self.mu.shape[0], i + 1, it, gain, self.name,
                    #    len(cep))
                #    break
                pass
            else:
                # logging.debug(
                #    'EM (start) distrib_nb: %d %i/%i llk: %f -- %s, %d',
                #    self.mu.shape[0], i + 1, it, llk[-1],
                #    self.name, len(cep))
                pass

        return llk

    def merge(self, model_list):
        """
        Merge a list of Mixtures into a new one. Weights are normalized uniformly
        :param model_list: a list of Mixture objects to merge
        """
        self.w = numpy.hstack(([mod.w for mod in model_list]))
        self.w /= self.w.sum()

        self.mu = numpy.vstack(([mod.mu for mod in model_list]))
        self.invcov = numpy.vstack(([mod.invcov for mod in model_list]))
        self.invchol = numpy.vstack(([mod.invchol for mod in model_list]))
        self.cov_var_ctl = numpy.vstack(([mod.cov_var_ctl for mod in model_list]))
        self.cst = numpy.hstack(([mod.cst for mod in model_list]))
        self.det = numpy.hstack(([mod.det for mod in model_list]))
        self.name = "_".join([mod.name for mod in model_list])
        self.A = numpy.hstack(([mod.A for mod in model_list]))

        self._compute_all()
        assert self.validate(), "Error while merging models"