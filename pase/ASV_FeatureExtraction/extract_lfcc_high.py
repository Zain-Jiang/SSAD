#!/usr/bin
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy import signal
import scipy.io as scio
#from librosa import feature

import fft_pre
import deltas

def extract_lfcc_high(speech, Fs, NFFT, No_Filter):
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

    # hz_bin = np.floor((NFFT + 1) * filBandwidthsf / Fs)

    # fbank = np.zeros((No_Filter, int(np.floor(NFFT / 2 + 1))))
    # for m in range(1, No_Filter + 1):
    #     f_m_minus = int(hz_bin[m - 1])   # left
    #     f_m = int(hz_bin[m])             # center
    #     f_m_plus = int(hz_bin[m + 1])    # right

    #     for k in range(f_m_minus, f_m):
    #         fbank[m - 1, k] = (k - hz_bin[m - 1]) / (hz_bin[m] - hz_bin[m - 1])
    #     for k in range(f_m, f_m_plus):
    #         fbank[m - 1, k] = (hz_bin[m + 1] - k) / (hz_bin[m + 1] - hz_bin[m])

    fbank = scio.loadmat('high_bank')
    fbank = fbank['bank']


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

    # t -= (np.mean(t, axis=0) + 1e-8)

    #####################DELTA and DDELTA############################

    # delta = signal.savgol_filter(t.T,3,1,deriv=1,delta = 1,axis = 1,mode='nearest').T
    # double_delta = signal.savgol_filter(delta.T,3,1, deriv=1,delta=1,axis=1,mode='nearest').T

    # delta = feature.delta(t)
    # double_delta = feature.delta(delta)

    # delta = np.diff(t.T,axis = 1).T
    # double_delta = np.diff(delta.T,axis = 1).T
    # return t[:-2],delta[:-1],double_delta
    # plt.pcolormesh(speech.T)
    # plt.show()
    delta = deltas(t.T,3).T
    double_delta = deltas(delta.T,3).T


    return t,delta,double_delta





if __name__ == '__main__':
    import soundfile as sf
    filepath = 'E:\\LA_T_1138215.flac'
    with open(filepath,'rb') as f:
        data,sr = sf.read(f)
    freqs, raw_fft = fft_pre(data,sr,0.028,0.014,512)
    stat,delta,double_delta = extract_lfcc_high(raw_fft,sr,512,20)
    cell = np.concatenate((stat,delta,double_delta),axis=1)
    print("hah")