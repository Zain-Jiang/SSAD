#!/usr/bin
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy import signal
import scipy.io as scio

def deltas(data,win_len=9):
    '''
    This function is used to calculate the deltas(derivatives) of a sequence use a window_len window.
    This is using a simple linear slope.
    Each row of data is filtered seperately.

    Input:
    data: the speech data to calculate delta
    win_len: the length of a window of the filter

    Output:
    delta: the  derivative of a sequence.

    Written by Li Zetian
    '''
    col,row = data.shape

    if(len(data) == 0):
        return [];
    else:

        #define window shape
        hlen = (int)(np.floor(win_len/2));
        win_len = 2 * hlen + 1;
        win = np.arange(hlen,-hlen-1, -1)

        #pad data by repeating first and last columns
        left_pad = np.repeat(data[:,0],hlen).reshape(-1,hlen)
        right_pad = np.repeat(data[:,row - 1],hlen).reshape(-1,hlen)
        pad_data = np.concatenate((left_pad,data,right_pad),axis=1)
        pad_data = np.where(pad_data == 0, np.finfo(float).eps,pad_data)

        #apply the delta filter
        delta = lfilter(win,1,pad_data,axis = 1,zi=None)

        #Trim edges
        selector = np.arange(0,row) + 2 * hlen
        delta = delta[:,selector]


        return delta

def fft_pre(speech, Fs, frame_size, frame_stride, NFFT):
    '''
    Function for get FFT result from an audio file
    Usage: freqs, rawFFT = fft_pre(speech,Fs,frame_size,frame_stride,NFFT)

    #Input:
    #speech: audio data
    #Fs: Sampling frequency in Hz
    #frame_size:length of a frame by seconds
    #frame_stride: length of the overlap by seconds
    #NFFT: number of FFT bins

    #Output:
    #freqs: series of frequency
    #rawFFT: Original fft result computing by axis = 1

    #Writen by Li Zetian

    #############################
    '''

    #####################PRE-EMPHASIS#################################
    pre_emphasis = 0.97
    speech =np.append(speech[0],speech[1:] - pre_emphasis*speech[:-1])

    #####################FRAMING######################################
    #frame_size帧长 frame_stride步长
    #frame_length：一帧对应的采样数
    #frame_step：一个步长对应的采样数
    frame_length, frame_step = frame_size * Fs, frame_stride * Fs
    signal_length = len(speech)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    #总帧数
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length + frame_step)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = int((num_frames - 1) * frame_step + frame_length)
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(speech, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    #####################WINDOWING####################################
    #frames *= np.blackman(frame_length)
    # plt.plot(x)
    # plt.show()
    frames *= np.hamming(frame_length)

    #####################FFT##########################################

    fft_frames =np.fft.rfft(frames, NFFT, axis = 1)
    freqs = np.linspace(0, Fs/2, NFFT/2+1)

    ##################################################################

    return freqs,fft_frames


def extract_lfcc(speech, Fs, NFFT, No_Filter):
    '''
    Function for computing LFCC features
    Usage: stat,delta,double_delta = extract_lfcc(speech,Fs,Window_length,No_Filter)

    #Input:
    #speech: speech,raw informaion of the speech file after fft
    #Fs: Sampling frequency in Hz
    #NFFT: number of fft bins
    #No_Filter: number of filter

    #Output:
    #stat: Static LFCC(Size: NxNo_Filter where N is the number of frames)
    #delta: Delta LFCC(Size: NxNo_Filter where N is the number of frames)
    #double_delta: Double Delta LFCC(Size: NxNo_Filter where N is the number of frames)

    Written by Li Zetian

    ##
    '''
    #####################POWER ENERGY################################

    speech = np.abs(speech)
    #speech = 20* np.log10(np.clip(np.abs(speech), 1e-8, 1e100))
    #speech = ((1.0 / NFFT) * ((np.abs(speech))**2))
    #speech = 20 * np.log10(np.abs(speech))

    #####################FILTERBANK##################################

    freqs = np.linspace(0, Fs/2, NFFT/2 + 1)
    filBandwidthsf = np.linspace(0, Fs/2, No_Filter+2)

    '''
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (Fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, No_Filter + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    '''

    hz_bin = np.floor((NFFT + 1) * filBandwidthsf / Fs)

    fbank = np.zeros((No_Filter, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, No_Filter + 1):
        f_m_minus = int(hz_bin[m - 1])   # left
        f_m = int(hz_bin[m])             # center
        f_m_plus = int(hz_bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - hz_bin[m - 1]) / (hz_bin[m] - hz_bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (hz_bin[m + 1] - k) / (hz_bin[m + 1] - hz_bin[m])
    filterbanks = np.dot(speech, fbank.T)
    filterbanks = np.where(filterbanks == 0, np.finfo(float).eps,filterbanks)
    filterbanks = np.log10(filterbanks)
    #####################Calculate Static Cepstral###################

    t = dct(filterbanks,type=2, axis=1,norm='ortho')[:,0:(No_Filter + 1)]

    #####################均值化优化###################################
    # plt.plot(fbank.T)
    # plt.show()
    # (nframes, ncoeff) = t.shape
    # n = np.arange(ncoeff)
    # cep_lifter=22
    # lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    # t *= lift  #*

    #t -= (np.mean(t, axis=0) + 1e-8)

    #####################DELTA and DDELTA############################

    # delta = signal.savgol_filter(t.T,3,1,deriv=1,delta = 1,axis = 1,mode='nearest').T
    # double_delta = signal.savgol_filter(delta.T,3,1, deriv=1,delta=1,axis=1,mode='nearest').T

    # delta = feature.delta(t)
    # double_delta = feature.delta(delta)

    # delta = np.diff(t.T,axis = 1).T
    # double_delta = np.diff(delta.T,axis = 1).T
    # return t[:-2],delta[:-1],double_delta

    # plt.pcolormesh(t.T,cmap=plt.get_cmap('YlOrRed'))

    # plt.colorbar()
    # plt.show()
    delta = deltas(t.T,3).T
    double_delta = deltas(delta.T,3).T


    return t,delta,double_delta





if __name__ == '__main__':
    import soundfile as sf
    filepath = 'E:\\00001.wav'
    with open(filepath,'rb') as f:
        data,sr = sf.read(f)
    freqs, raw_fft = fft_pre(data,sr,0.016,0.008,256)
    stat,delta,double_delta = extract_lfcc(raw_fft,sr,256,20)
    cell = np.concatenate((stat,delta,double_delta),axis=1)
    print("hah")