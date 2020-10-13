# -*- coding:utf-8 -*-

import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt


from featureExtraction.fft_pre import fft_pre
from featureExtraction.deltas import deltas

def extract_rps(speech, Fs, NFFT, No_Filter):
    '''
    This function is used to computing the Relative Phase Spectrum features.

    #Input:
    #speech: speech,raw informaion of the speech file after fft
    #Fs: Sampling frequency in Hz
    #NFFT: number of fft bins
    #No_Filter: number of filter

    #Output:
    #stat: static Relative phase spectrum features 0-axis is along time
    #delta: first derive of stat
    #double_delta: second derive of stat

    ####
    '''

    #########################PHASE INFORMATION########################
    speech = np.angle(speech)
    speech = np.unwrap(speech) ###avoid phase wrap
    speech = np.diff(speech)
    #########################FILTERBANK###############################
    freqs = np.linspace(0, Fs/2, NFFT/2 + 1)
    filBandwidthsf = np.linspace(0, Fs/2, No_Filter+2)

    hz_bin = np.floor((NFFT+1) * filBandwidthsf/ Fs)

    fbank = np.zeros((No_Filter, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, No_Filter + 1):
        f_m_minus = int(hz_bin[m - 1])   # left
        f_m = int(hz_bin[m])             # center
        f_m_plus = int(hz_bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - hz_bin[m - 1]) / (hz_bin[m] - hz_bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (hz_bin[m + 1] - k) / (hz_bin[m + 1] - hz_bin[m])
    filterbanks = np.dot(speech, fbank[:,:-1].T)
    filterbanks = np.where(filterbanks == 0, np.finfo(float).eps,filterbanks)
    filterbanks = np.log10(np.clip(filterbanks,1e-8,1e100))

    #####################Calculate Static Cepstral###################

    t = dct(filterbanks,type=2, axis=1,norm='ortho')[:,0:(No_Filter + 1)]

    #####################DELTA and DDELTA############################

    delta = deltas(t.T,3).T
    double_delta = deltas(delta.T,3).T

    # plt.pcolormesh(speech.T)
    # plt.show()
    # plt.pcolormesh(t.T)
    # plt.show()
    # plt.pcolormesh(delta.T)
    # plt.show()
    # plt.pcolormesh(double_delta.T)
    # plt.show()
    return t,delta,double_delta




if __name__ == '__main__':
    import soundfile as sf
    filepath = 'E:\\LA_T_1138215.flac'
    with open(filepath,'rb') as f:
        data,sr = sf.read(f)
    freqs, raw_fft = fft_pre(data,sr,0.028,0.014,512)
    stat,delta,double_delta = extract_rps(raw_fft,sr,512,20)
    cell = np.concatenate((stat,delta,double_delta),axis=1)
    print("hah")