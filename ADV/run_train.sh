#!/bin/sh

path=D:/AdvAttacksASVspoof/_saved/models/LA_SENet12_LPSseg_uf_seg600/20200412_094534/checkpoint-epoch6.pth
configPath1=_configs/config_LA_SENet12_LPSseg_uf_seg600.json
#configPath2=D:/AdvAttacksASVspoof/_saved\models/LA_SENet12_LPSseg_uf_seg600/20200412_094534/config.json
GPU1=1

python train.py  --config ${configPath1} --device ${GPU1}   #> /dev/null 2>&1 &
