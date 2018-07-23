# Created on 2018-07-23 (Author: Kaituo Xu)
# Based on Kaldi Fbank
# Based on github repo:
# - https://github.com/ZitengWang/python_kaldi_features
# - https://github.com/jameslyons/python_speech_features

import decimal
import numpy
import math

def kaldi_fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
        nfilt=40, round_to_power_of_two=True, lowfreq=20, highfreq=None,
        dither=1.0, remove_dc_offset=True, preemph=0.97, wintype='hamming'):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 40.
    :param round_to_power_of_two: compute nfft, default True
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 20.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param wintype: hamming or povey
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features.
        Each row holds 1 feature vector. The second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    window_size = winlen * samplerate
    nfft = round_to_nearest_power_of_two(window_size) if round_to_power_of_two else window_size
    frames,raw_frames = framesig(signal, winlen*samplerate, winstep*samplerate, dither, preemph, remove_dc_offset, wintype)
    pspec = powspec(frames,nfft) # nearly the same until this part
    energy = numpy.sum(raw_frames**2,1) # this stores the raw energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy

def kaldi_logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
        nfilt=40, round_to_power_of_two=True, lowfreq=20, highfreq=None,
        dither=1.0, remove_dc_offset=True, preemph=0.97, wintype='hamming'):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param round_to_power_of_two: compute nfft, default True
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 20.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param wintype: hamming or povey
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal, samplerate, winlen, winstep,
                        nfilt, round_to_power_of_two, lowfreq, highfreq,
                        dither, remove_dc_offset, preemph, wintype)
    return numpy.log(feat)

# TODO: make this same with kaldi add-deltas
def delta(feat, N=2):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat

# --- utils ---

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 1127 * numpy.log(1+hz/700.0)

def get_filterbanks(nfilt=26,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)

    # check kaldi/src/feat/Mel-computations.h    
    fbank = numpy.zeros([nfilt,nfft//2+1])
    mel_freq_delta = (highmel-lowmel)/(nfilt+1)
    for j in range(0,nfilt):
        leftmel = lowmel+j*mel_freq_delta
        centermel = lowmel+(j+1)*mel_freq_delta
        rightmel = lowmel+(j+2)*mel_freq_delta
        for i in range(0,nfft//2):
            mel=hz2mel(i*samplerate/nfft)
            if mel>leftmel and mel<rightmel:
                if mel<centermel:
                    fbank[j,i]=(mel-leftmel)/(centermel-leftmel)
                else:
                    fbank[j,i]=(rightmel-mel)/(rightmel-centermel)
    return fbank

# --- from sigproc.py ---

def round_to_nearest_power_of_two(window_size):
    n = 1
    while n < window_size:
        n *= 2
    return n

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

def framesig(sig, frame_len, frame_step, dither=1.0, preemph=0.97, remove_dc_offset=True, wintype='hamming', stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        # numframes = 1 + (( slen - frame_len) / frame_step)
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    # check kaldi/src/feat/feature-window.h
    padsignal = sig[:(numframes-1)*frame_step+frame_len]
    if wintype is 'povey':
        win = numpy.empty(frame_len)
        for i in range(frame_len):
            win[i] = (0.5-0.5*numpy.cos(2*numpy.pi/(frame_len-1)*i))**0.85     
    else: # the hamming window
        win = numpy.hamming(frame_len)
        
    if stride_trick:
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(win, (numframes, 1))
        
    frames = frames.astype(numpy.float32)
    raw_frames = numpy.zeros(frames.shape)
    for frm in range(frames.shape[0]):
        frames[frm,:] = do_dither(frames[frm,:], dither)        # dither
        frames[frm,:] = do_remove_dc_offset(frames[frm,:])      # remove dc offset
        raw_frames[frm,:] = frames[frm,:]
        frames[frm,:] = do_preemphasis(frames[frm,:], preemph)    # preemphasize

    return frames * win, raw_frames

def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)

def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return numpy.square(magspec(frames, NFFT))

def do_dither(signal, dither_value=1.0):
    signal += numpy.random.normal(size=signal.shape) * dither_value
    return signal
    
def do_remove_dc_offset(signal):
    signal -= numpy.mean(signal)
    return signal

def do_preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append((1-coeff)*signal[0], signal[1:] - coeff * signal[:-1])


if __name__ == "__main__":
    import scipy.io.wavfile as wav
    rate, sig = wav.read("english.wav")
    fbank_feat = logfbank(sig, samplerate=rate, dither=0, wintype="hamming")
    print(fbank_feat)