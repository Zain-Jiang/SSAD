# Self-supervised Spoofing Audio Detection Scheme
<u>This repository contains the implementations of SSAD</u>，which is a speech waveform encoder trained in a self-supervised manner with the so called worker framework. A SSAD model can be used as a speech feature extractor or a pre-trained encoder for our spoofing audio detection task.

## Requirements
- PyTorch 1.0 or higher
- Torchvision 0.2 or higher
- Install the requirements from `requirements.txt`: `pip install -r requirements.txt`
- Intall SSAD modules python: `setup.py install`

*NOTE: Edit the cupy-cuda100 requirement in the file if needed depending on your CUDA version. Defaults to 10.0 now*

```bash
#use this phrase to set the root path whenever you start
export PYTHONPATH=.
```

## Pre-trained Model
This is the **valid_loss** in tensorboard:

<img src="https://github.com/DangerousQiang/SSAD/blob/main/images/valid_loss.png" alt="image-20201013001612361" style="zoom:30%;" />

The pretrained SSAD encoder model has been trained for the 99 epochs (in my experience, you can trained it for 200 epochs for best appearance), and the classifier is casually trained by me. The results and URLs are as follows.

| min-tDCF | EER      | classifier's name |
| -------- | -------- | ----------------- |
| 0.1643   | 6.3869 % | SENet12           |

URLs: https://pan.baidu.com/s/1fx-fk2rOPWMNgLRlwJ33JA  passcode: [4j1b]()



**<u>Remember to modify the path and config in the bash scripts below.</u>**

## Data preparation
To make the data preparation the following files have to be provided:

- training files list `train_scp`: contains a file name per line (without directory names), including `.wav`/`mp3`/etc. extension.

- test files list `test_scp`: contains a `wav` file name per line (without directory names), including `.wav`/`mp3`/etc. extension.

- dictionary with filename `dict.npy`-> integer speaker class (speaker id) correspondence (same filenames as in train/test lists).

- `data.cfg`: the dataset config file

- `stats_pase+.pkl`: the normalization statistics for the workers to work properly

  ```bash
  sbatch -A yzren -p gpu --gres=gpu:1 -c 16 preprocess/preprocess.sh
  ```

  #time: about 50 minutes

## Train SSAD encoder
```bash
sbatch -A yzren -p gpu --gres=gpu:1 -c 16 train.sh
```

#time: about 2 days for 100 epochs

## Extract features for classifier
```bash
sbatch -A yzren -p gpu --gres=gpu:1 -c 16 feature.sh
```

## Train classifier
Operations for classifier should be made in the directory named ADV, and you should modify the config file: `ADV/_configs/config_LA_SENet12_LPSseg_uf_seg600.json`

```bash
cd ADV
sbatch -A yzren -p gpu --gres=gpu:1 -c 16 run_train.sh
```

## Evaluation
```bash
sbatch -A yzren -p gpu --gres=gpu:1 -c 16 run_eval.sh
```

