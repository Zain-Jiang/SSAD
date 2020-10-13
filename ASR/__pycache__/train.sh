#!/bin/bash

python -u  train.py --batch_size 16 --epoch 400 --save_path pase+_ckpt --num_workers 4 --warmup 10000000 --net_cfg cfg/workers/workers+.cfg --fe_cfg cfg/frontend/PASE+.cfg --data_cfg data/librispeech_data.cfg --min_lr 0.0005 --fe_lr 0.001 --data_root /home/jiangziyue/dat01/LA/ASVspoof2019_LA_train/flac --stats data/ASVspoof2019_stats_pase+.pkl  --tensorboard False --backprop_mode base  --lr_mode poly
