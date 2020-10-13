#!/bin/sh

#prepare list of training, dev (full dev or four fold)
RootDir=`pwd`
echo 'Current Dir: '${RootDir}

extractLPS=1
if [ $extractLPS -eq 1 ]; then
    echo 'Extracting Log Power Spectrograms ...'
    python ./preprocess/compute_LPS.py --in_dir /home/jiangziyue/dat01 \
                                       --out_dir /home/jiangziyue/dat01/adv/data_LA/LPS_t45 \
                                       --access_type LA \
                                       --param_json_path ./preprocess/conf/stft_T45.json
fi


extractLMS=0
if [ $extractLMS -eq 1 ]; then
    echo 'Extracting Log Magnitude Spectrograms ...'
    python ./preprocess/compute_LMS.py --in_dir E:\\ASVspoof\\ASVspoof_root \
                                       --out_dir ./data/LMS \
                                       --access_type PA \
                                       --param_json_path ./preprocess/conf/stft.json
fi

