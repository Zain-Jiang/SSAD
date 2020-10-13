#!/bin/sh

GPU=0
#checkpoint-epoch1.pth
#model_best.pth
python interface.py  --resume _saved_PA/models/LA_SENet12_LPSseg_uf_seg600/20200731_171931/model_best.pth \
                    --protocol_file /home/jiangziyue/dat01/pase/data/XinAn/result/protocol.txt \
                    --asv_score_file data_LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt \
                    --device ${GPU}    