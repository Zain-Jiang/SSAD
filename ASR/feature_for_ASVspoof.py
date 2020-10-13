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

modes=["train","dev","eval"]

for mode in modes:
    pase_cfg = "cfg/frontend/PASE+_tcn_projection.cfg"  # e.g, '../cfg/frontend/PASE.cfg'
    pase_model = "ASV_ckpt/FE_e99.ckpt"  # e.g, '../FE_e199.ckp' (download the pre-trained PASE+ model as described in the doc)
    data_folder = "/home/jiangziyue/dat01/LA/ASVspoof2019_LA_"+mode  # e.g., '/home/mirco/Dataset/TIMIT'
    output_root = "/home/jiangziyue/dat01/pase_feature_LA"

    protocol_file_train='/home/jiangziyue/dat01/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    protocol_file_dev="/home/jiangziyue/dat01/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    protocol_file_eval="/home/jiangziyue/dat01/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

    if mode=="train":
        protocol_file=protocol_file_train
    elif mode=="dev":
        protocol_file=protocol_file_dev
    elif mode=="eval":
        protocol_file=protocol_file_eval
    else:
        print("mode error.")

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


    protocol_list = [line.rstrip('\n') for line in open(protocol_file)]

    #get the name and label of wav file
    wavname_list=[]
    label_list=[]
    for item in protocol_list:
        item=item.split(" ")
        if item[-1]=="spoof":
            label_list+=[0]
        elif item[-1]=="bonafide":
            label_list+=[1]
        wavname_list.append(item[1])
        
    assert len(wavname_list)==len(label_list)
    # print(wavname_list)
    # print(label_list)

    #split to save memory
    f = lambda a:map(lambda b:a[b:b+1000],range(0,len(a),1000))
    wavname_list=f(wavname_list)

    count=0
    for batch in tqdm(wavname_list):
        # reading the training signals
        print("Waveform reading...")
        fea={}
        type="flac"
        for item in batch:
            [signal, fs] = sf.read(data_folder+'/'+type+'/'+item+"."+type)
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
                np.save(output_root+"/asv_"+"train"+"/"+snt_id+".npy",fea_pase[snt_id])
                # print('Processed training utterance {}/{} features'.format(wi + 1,
                                                                           # len(fea.keys())))

        inp_dim=fea_pase[snt_id].shape[1]*(left+right+1)
        count+=1

