#!/bin/bash

LA_path="/home/jiangziyue/dat01/LA"


python -u  train.py --batch_size 20 --epoch 200 --save_path ASV_ckpt \
	       --num_workers 16 --warmup 10000000 --net_cfg cfg/workers/workers+.cfg \
	       --fe_cfg cfg/frontend/PASE+_tcn_projection.cfg \
	       --data_cfg data/ASVspoof2019/ASVspoof2019_data.cfg \
	       --min_lr 0.0005 --fe_lr 0.001 \
	       --data_root "${LA_path}/ASVspoof2019_LA_train/flac/" \
	       --stats data/ASVspoof2019/ASVspoof2019_stats_pase+.pkl \
	       --chunk_size 32000 \
	       --tensorboard True \
	       --backprop_mode base\
	       --random_scale True\
	       --lr_mode poly