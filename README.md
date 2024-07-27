# Spatio-Temporal Dual-Attention Transformer for Time-Series Behavioral Biometrics
This repository contains code for our BehaveFormer framework on behaviour biometrics continuous identification using swipe data (scroll up and scroll down gestures), published in IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM, Q1 Journal) [\[IEEE link\]](https://ieeexplore.ieee.org/document/10510407) [\[free download\]](https://www.dropbox.com/scl/fi/5lxfacb7e9zj7019ncqp6/TBIOM3394875.pdf?rlkey=et3ryhn1baminwahjw2wyu773&dl=0).

We extend the original BehaveFormer module built for keystroke data at [this repo](https://github.com/dilshansenarath/behaveformer) with several major changes: 
- Allow for easier adaptation to new modality or dataset, change to process data on the fly during training to accommodate for large datasets.
- Faster preprocessing for time series data (e.g. IMU synchronization): For example, on HuMIdb dataset, the processing speed using 8 CPUs reduces from 3 hours to 5 minutes (a 36x reduction). 
- Training procedures (e.g., optimizer, learning rate, etc.) are changed to achieve better performance. Please check the paper for details.
- Allow for datasets with varying numbers of sessions per user. In this case, during verification, you only need to declare the number of enrollment sessions, the rest of the available sessions will be used for verification.

Even though this repo focuses on swipe data, the framework can easily be extended to other modalities with little effort.

## Abstract
Continuous Authentication (CA) using behavioral biometrics is a type of biometric identification that recognizes individuals based on their unique behavioral characteristics. Many behavioral biometrics can be captured through multiple sensors, each providing multichannel time-series data. Utilizing this multichannel data effectively can enhance the accuracy of behavioral biometrics-based CA. This paper extends BehaveFormer, a new framework that effectively combines time series data from multiple sensors to provide higher security in behavioral biometrics. BehaveFormer includes two Spatio-Temporal Dual Attention Transformers (STDAT), a novel transformer we introduce to extract more discriminative features from multichannel time-series data. Experimental results on two behavioral biometrics, Keystroke Dynamics and Swipe Dynamics with Inertial Measurement Unit (IMU), have shown State-of-the-art performance. For Keystroke, on three publicly available datasets (Aalto DB, HMOG DB, and HuMIdb), BehaveFormer outperforms the SOTA. For instance, BehaveFormer achieved an EER of 2.95\% on the HuMIdb. For Swipe, on two publicly available datasets (HuMIdb and FETA) BehaveFormer outperforms the SOTA, for instance, BehaveFormer achieved an EER of 3.67\% on the HuMIdb. Additionally, the BehaveFormer model shows superior performance in various CA-specific evaluation metrics. The proposed STDAT-based BehaveFormer architecture can also be effectively used for transfer learning. The model weights and reproducible experimental results are available at [this repo]().

## Installation
We use Python 3.10.12 and Pytorch 2.0.1 for our experiments. 
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
Example command to run training on HuMIdb with scroll down and all IMU data. For other datasets and experiments, please change the `--dataname` and `-c`:
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
We include the best model weights for each experiment in [this folder](https://www.dropbox.com/scl/fi/sy4p1eu25xgh6nxu7ckdv/best_weights.zip?rlkey=i1tf5croix3r9okumd6pfqew7&st=zanaw9q3&dl=0). After download, extract the zip file to the same code folder by running: `unzip best_weights.zip`.

## Updates
- 2024-05-01: Release pre-trained weights
- 2023-11-18: Initial code release

## References
[1]: A. Acien, A. Morales, J. Fierrez, R. Vera-Rodriguez, and O. Delgado-Mohatar, “Becaptcha: Behavioral bot detection using touchscreen and mobile sensors benchmarked on humidb,” Engineering Applications of Artificial Intelligence, vol. 98, p. 104058, 2021.

[2]: M. Georgiev, S. Eberz, H. Turner, G. Lovisotto, and I. Martinovic, “Feta: Fair evaluation of touch-based authentication,” 2022. \[Online\]. Available: https://api.semanticscholar.org/CorpusID:255546212
