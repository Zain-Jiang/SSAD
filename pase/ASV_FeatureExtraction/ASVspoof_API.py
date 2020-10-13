# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:50:31 2020

@author: jzy
"""

import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from scipy.signal import lfilter
from scipy import signal
import resampy
import scipy
import scipy.io as scio
import os

def fft_pre(speech, Fs, frame_size, frame_stride, NFFT):

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
    freqs = np.linspace(0, int(Fs/2), int(NFFT/2+1))

    ##################################################################

    return freqs,fft_frames

def deltas(data,win_len=9):
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

def extract_lfcc(speech, Fs, NFFT, No_Filter):

    #####################POWER ENERGY################################

    speech = np.abs(speech)
    #speech = 20* np.log10(np.clip(np.abs(speech), 1e-8, 1e100))
    #speech = ((1.0 / NFFT) * ((np.abs(speech))**2))
    #speech = 20 * np.log10(np.abs(speech))

    #####################FILTERBANK##################################

    freqs = np.linspace(0, int(Fs/2), int(NFFT/2 + 1))
    filBandwidthsf = np.linspace(0, int(Fs/2), int(No_Filter+2))

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

    delta = deltas(t.T,3).T
    double_delta = deltas(delta.T,3).T

    return t,delta,double_delta


def extract_cqcc(wav_path):
    
    with open(wav_path,'rb') as f:
        data,sr = sf.read(f)
        stat=librosa.cqt(data, sr=sr)
        delta=deltas(stat.T,3).T
        double_delta=deltas(delta.T,3).T
        cell = np.concatenate((stat,delta,double_delta),axis=1)
        return cell
    
def extract_lfcc_highfreq(wav_path):
    with open(wav_path,'rb') as f:
        data,sr = sf.read(f)
        freqs, raw_fft = fft_pre(data,sr,0.035,0.030,512)
        stat,delta,double_delta = extract_lfcc_high(raw_fft,sr,512,20)
        cell = np.concatenate((stat,delta,double_delta),axis=1)
        return cell.T

def extract_cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD):
    CQT = librosa.cqt(y=x,sr=fs,bins_per_octave=B,fmin=fmin,norm=0.625,n_bins=863,hop_length=128)

    ##OG POWER SPECTRUM
    LogP_absCQT = np.log(np.square(np.abs(CQT)) + 2.2204e-16)
    
    ##UNIFORM RESAMPLING
    Ures_LogP_absCQT=resampy.resample(LogP_absCQT, 938,100,axis=0)
    
    # DCT
    CQcepstrum = dct(Ures_LogP_absCQT,axis=0)
    
    #DYNAMIC COEFFICIENTS
    CQcepstrum_temp = CQcepstrum[0:cf,:]
    f_d = 3; # delta window size
    CQcepstrum_temp_delta1 = deltas(CQcepstrum_temp,f_d)
    CQcepstrum_temp_delta2 = deltas(CQcepstrum_temp_delta1,f_d)

    CQCC = np.concatenate((CQcepstrum_temp,CQcepstrum_temp_delta1,CQcepstrum_temp_delta2),axis=0)
    # print(CQcepstrum_temp.shape)
    # print(CQcepstrum_temp_delta1.shape)
    # print(CQcepstrum_temp_delta2.shape)
    # print(CQCC.shape)
    
    # plt.plot(CQCC,linewidth = 0.5)
    # plt.show()
    return CQCC




if __name__ == "__main__":
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
    wavfile="E:/ASVspoof/ASVspoof_root/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac"
    x,fs=sf.read(wavfile)
    freqs, raw_fft = fft_pre(x,16000,0.025,0.015,512)
    cqcc=extract_cqcc(x[0:16000], 16000, 96, fs/2, fs/2**10, 16, 40, 'ZsdD')
    stat,delta,double_delta = extract_lfcc(raw_fft,16000,512,40)
    lfcc = np.concatenate((stat,delta,double_delta),axis=1)
    
    print(cqcc.T.shape)
    print(lfcc.shape)



