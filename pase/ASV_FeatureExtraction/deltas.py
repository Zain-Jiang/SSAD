import numpy as np
from scipy.signal import lfilter

import fft_pre


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



if __name__ == '__main__':
    import soundfile as sf
    filePath = 'E:\\LA_T_1138215.flac'
    with open(filePath,'rb') as f:
        data, sr = sf.read(f)
    freqs, raw_fft = fft_pre(data,sr,0.020,0.010,512)

    print('Done!')