#!/bin/bash

##################################################################################################
# Script to run ablation study on HuMIdb: 
# Example: 
# sh scripts/humi/ablation/train.sh <down/up> <acc/gyr/mag/acc-gyr/acc-mag/gyr-mag> <0 (optional)>
##################################################################################################

action=$1  # down, up
imu_type=$2  # acc, gyr, mag, acc-gyr, acc-mag, gyr-mag
cuda_device=$3  # 0 if not passed in

# Check input
if [[ $action != 'down' && $action != 'up' ]]; then
    echo "Wrong action, must be down or up"
fi
if [[ $imu_type != 'acc' && $imu_type != 'gyr' && $imu_type != 'mag' && $imu_type != 'acc-gyr' && $imu_type != 'acc-mag' && $imu_type != 'gyr-mag' ]]; then
    echo "Wrong imu_type, must be: acc, gyr, mag, acc-gyr, acc-mag, gyr-mag"
fi
if [[ -z $cuda_device ]]; then
    cuda_device=0
fi

# Main work
echo "Running with scroll ${action} and ${imu_type}"

config_file="configs/humi/ablation/humi_scroll50${action}_imu100${imu_type}_epoch500_enroll3_b128.yaml"
cmd="CUDA_VISIBLE_DEVICES=${cuda_device} python train.py --dataname humi -c ${config_file} --work_dirs_subfolder ablation/humi --checkpoint_interval -1"
echo $cmd
eval $cmd