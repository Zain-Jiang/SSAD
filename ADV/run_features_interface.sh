#prepare list of training, dev (full dev or four fold)
RootDir=`pwd`
echo 'Current Dir: '${RootDir}

extractLPS=1
if [ $extractLPS -eq 1 ]; then
    echo 'Extracting Log Power Spectrograms ...'
    python ./preprocess/compute_LPS_interface.py --in_dir ./uploaded_wav \
                                       --out_dir ./data_LA/LPS_interface \
                                       --access_type LA \
                                       --param_json_path ./preprocess/conf/stft_T45.json
fi

