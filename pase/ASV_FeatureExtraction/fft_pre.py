# -*- coding:utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt

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





if __name__ == '__main__':
    import soundfile as sf

    filepath = 'E:\\ASVSPOOF\\ASVspoof_root\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1000406.flac'
    with open(filepath,'rb') as f:
        data,sr = sf.read(f)
    freqs,raw_fft = fft_pre(data,sr,0.025,0.010,512)
