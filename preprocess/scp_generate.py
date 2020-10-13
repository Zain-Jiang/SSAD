#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
from sklearn.model_selection import train_test_split  # 导入train_test_split用于数据分割
import numpy as np
import glob


def load_data_path(path):
    path_list = glob.glob(path)
    name_list = []
    for i in path_list:
        name_list.append(i.split("\\")[-1])
    name_list.sort()
    return name_list


def load_label(path):
    protocol_file = path
    protocol_list = [line.rstrip('\n') for line in open(protocol_file)]

    list_cache = []
    for item in protocol_list:
        item = item.split(" ")
        item = item[1] + " " + item[-2] + " " + item[-1]
        list_cache.append(item)
    protocol_list = list_cache
    protocol_list.sort()

    return protocol_list


def main(argv):
    root_path_ASVspoof2019=argv[0]
    print(root_path_ASVspoof2019)

    # 1.数据获取 + scp文件生成
    dir_path = str(root_path_ASVspoof2019)+"/ASVspoof2019_LA_train/flac/"
    audio_names = load_data_path(dir_path+"*.flac")
    bad_label = [0] * (len(audio_names))

    X_train, X_test, y_train, y_test = train_test_split(audio_names, bad_label, test_size=0.10, random_state=33)

    print("find " + str(len(audio_names)) + " audios in total, the 1-10 is: ")
    print(audio_names[0:10])

    with open('data/ASVspoof2019/ASVspoof2019_tr.scp', 'w') as f:
        for item in X_train:
            f.write(item + "\n")

    with open('data/ASVspoof2019/ASVspoof2019_te.scp', 'w') as f:
        for item in X_test:
            f.write(item + "\n")

    # 2.标签获取 + dict.npy文件生成
    protocol_file=str(root_path_ASVspoof2019)+"/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    labels = load_label(protocol_file)
    dic = dict()
    for item in labels:
        item = item.split(" ")
        if item[-1] == "bonafide":
            dic[dir_path+item[0] + ".flac"] = 0
        # elif item[-2][0]=="A":
        # dic[item[0]+".flac"]=int(item[-2][2])
        else:
            dic[dir_path+item[0] + ".flac"] = 1
    np.save("ASVspoof2019_dict.npy", dic)
    print("find " + str(len(dic)) + " dict item in total, the 1-10 is: ")
    count=0
    for i in dic:
        print("dict[%s]=" % i, dic[i])
        count+=1
        if count>10:
            break

    np.save("data/ASVspoof2019/ASVspoof2019_dict.npy", dic)

    print("scp file generation is Done---------------------------------------------------")


if __name__ == "__main__":
    main(sys.argv[1:])
