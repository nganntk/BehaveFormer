# Spatio-Temporal Dual-Attention Transformer for Time-Series Behavioral Biometrics
This repository contains code for the BehaveFormer framework with on behaviour biometrics continuous identification using swipe data (scroll up and scroll down gestures). 
For BehaveFormer using keystroke data, please check this [repo](https://github.com/dilshansenarath/behaveformer), which the code in this repo is based on.

## Abtract
To be updated

## Installation
We use python 3.10.12 and Pytorch 2.0.1 for our experiments. 
Please use the following commands to install the environment for this project:

```bash
conda create -n BehaveFormer python==3.10.12
conda activate BehaveFormer

bash install.sh
```

## Datasets
### Download data
| Dataset | Subjects | Sessions | Actions | Modalities | Download |
|:---:|:---:|:---:|---|---|---|
| Humidb [1] | 599 | 1-5 | Typing, Swipe, Tap, Hand gesture, Finger writing | Keystroke, Touch, IMU, Light, GPS, WiFi, Bluetooth, Orientation, Proximity, Microphone | [Link](https://github.com/BiDAlab/HuMIdb) |
| FETA [2] | 470 | 1-31 | Scrolling social media, Browsing image gallery | Touch, IMU |  [Link](https://github.com/ssloxford/evaluation-pitfalls-touch) |

### Preprocessing
This step extracts the scroll features and synchronize IMU and scroll data. We use multiple CPUs during preprocessing, if you want to run on 1 cpu, remove `--ncpus` argument.

For HuMIdb, we split the data to train, validation and test sets before preprocessing. The preprocessed data will be saved to 3 files `XXX_scroll_imu_data_all.pickle`, with `XXX` can be `training`, `validation`, or `testing`. However, for FETA, we saved the processed data for all users, then get the data split. Note that for FETA, we only include max 10 sessions per user, with 1 repetition with scroll down and scroll up data per session. 
To run the preprocessing step, please update the paths in `dataset_root` and `output_dir` in the config file, then run the following commands to process HuMIdb and FETA.

- To preprocess HuMIdb with 8 cpus
```
python -m preprocessing.preprocess_humidb --config "configs/humi/preprocess/humi_scroll50downup_imu100.yaml" --ncpus 8
```
- To preprocess FETA with 16 cpus, run the following lines
```
# Run this to process data for all users
python -m preprocessing.preprocess_feta --config "configs/feta/preprocess/feta_scroll50downup_imu100.yaml" --mode "preprocess" --ncpus 16

# Run this to get the data split
python -m preprocessing.preprocess_feta --config "configs/feta/preprocess/feta_scroll50downup_imu100.yaml" --mode "split"
```

## Code and usage
Example command to run training on HuMIdb with scroll down and all IMU data. For other dataset and experiment, please change the `--dataname` and `-c`:
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataname humi -c configs/humi/main/humi_scroll50down_imu100all_epoch500_enroll3_b128.yaml
```

Alternatively, we also provide wrapper scripts in `scripts` folder for all the experiments in our paper. To run a specific experiment, you can use the following command format:
```
sh scripts/<dataname>/<exp_set>/<step>.sh <action> <imu_type> <cuda_device, optional> 
```
- `dataname`: `humi` or `feta`
- `exp_set`: `main` or `ablation`
- `step`: `train` or `test`
- `action`: `down` or `up`
- `cuda_device` (optional): for example, `0`
- For `imu_type`, the following values are allowed based on `dataname` and `exp_set`

| dataname | exp_set | imu_type |
|---|---|---|
| humi | main | all, none |
| humi | ablation | acc, gyr, mag, acc-gyr, acc-mag, gyr-mag |
| feta | main | all, none |
| feta | ablation | acc, gyr |

## Model weights
We include the best model weights for each experiment in [this folder](). After download, extract the zip file to the same code folder by running: `unzip best_weights.zip`.

## Updates
- 2023-11-18: Initial code release

## References
[1]: A. Acien, A. Morales, J. Fierrez, R. Vera-Rodriguez, and O. Delgado-Mohatar, “Becaptcha: Behavioral bot detection using touchscreen and mobile sensors benchmarked on humidb,” Engineering Applications of Artificial Intelligence, vol. 98, p. 104058, 2021.

[2]: M. Georgiev, S. Eberz, H. Turner, G. Lovisotto, and I. Martinovic, “Feta: Fair evaluation of touch-based authentication,” 2022. \[Online\]. Available: https://api.semanticscholar.org/CorpusID:255546212