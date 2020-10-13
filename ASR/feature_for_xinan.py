# ======================================================================

        # Written by Ziyue Jiang.   
        # All rights reserved

        # filename : feature_for_ASVspoof.py
        # description :

        # created at  04/03/2020 02:34:17
        # Whu university

# ======================================================================


import warnings
warnings.filterwarnings("ignore")

import librosa
import os
import sys
from neural_networks import MLP,context_window
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
from pase.models.frontend import wf_builder
import soundfile as sf
import os
import json
from pase.models.WorkerScheduler.encoder import *




pase_cfg = "../cfg/frontend/PASE+_tcn_projection.cfg"  # e.g, '../cfg/frontend/PASE.cfg'
pase_model = "/home/jiangziyue/dat01/pase/XinAn_ckpt/FE_e799.ckpt"  # e.g, '../FE_e199.ckp' (download the pre-trained PASE+ model as described in the doc)
data_folder = "/home/jiangziyue/dat01/XinAn/audio"  # e.g., '/home/mirco/Dataset/TIMIT'
output_file='TIMIT_asr_exp.res' # e.g., 'TIMIT_asr_exp.res'


# Training parameters
left=1
right=1

#device=0 #get_freer_gpu()
device='cuda:0'

# Loading pase
pase =wf_builder(pase_cfg)
pase.load_pretrained(pase_model, load_last=True, verbose=False)
pase.to(device)
pase.eval()

f = open('/home/jiangziyue/dat01/XinAn/result/protocol.txt','w',encoding='utf-8')

for item in tqdm(os.listdir(data_folder)):
    # reading the training signals
    print("Waveform reading...")
    fea={}
    type="flac"

    [signal, fs] = sf.read(data_folder+'/'+item)
    signal=signal/np.max(np.abs(signal))
    signal = signal.astype(np.float32)
        
    fea_id=item
    fea[fea_id]=torch.from_numpy(signal).float().view(1,1,-1)


    # Computing pase features for training
    print('Computing PASE features...')
    with torch.no_grad():
        fea_pase={}
        for wi, snt_id in enumerate(fea.keys()):
            #pase.eval()
            fea_pase[snt_id]=pase(fea[snt_id].to(device), device=device).to('cpu').detach()
            fea_pase[snt_id]=fea_pase[snt_id].view(fea_pase[snt_id].shape[1],fea_pase[snt_id].shape[2]).transpose(0,1)
            np.save("/home/jiangziyue/dat01/XinAn/result/"+snt_id+".npy",fea_pase[snt_id])
            if snt_id[0]=="R":
                f.write("LA_0069 "+snt_id+" - - bonafide")
            else:
                f.write("LA_0069 " + snt_id + " - - spoof")
            f.write("\n")

f.close()