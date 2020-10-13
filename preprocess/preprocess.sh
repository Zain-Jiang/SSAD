#!/bin/bash

LA_path="/home/jiangziyue/dat01/LA"

python preprocess/scp_generate.py ${LA_path}

python unsupervised_data_cfg_librispeech.py \
  --data_root /home/jiangziyue/dat01/pase/data/XinAn/wav/ \
	--train_scp data/ASVspoof2019/ASVspoof2019_tr.scp \
	--test_scp data/ASVspoof2019/ASVspoof2019_te.scp \
	--libri_dict data/ASVspoof2019/ASVspoof2019_dict.npy \
	--cfg_file data/ASVspoof2019/ASVspoof2019_data.cfg

python make_trainset_statistics.py --data_root "${LA_path}/ASVspoof2019_LA_train/flac/" \
	--data_cfg data/ASVspoof2019/ASVspoof2019_data.cfg \
	--net_cfg cfg/workers/workers+.cfg \
	--out_file data/ASVspoof2019/ASVspoof2019_stats_pase+.pkl