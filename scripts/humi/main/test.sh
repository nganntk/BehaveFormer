#!/bin/bash

############################################################################
# Script to run main results on HuMIdb: 
# Example: sh scripts/humi/main/test.sh <down/up> <all/none> <0 (optional)>
############################################################################

action=$1  # down, up
imu_type=$2  # all, none
cuda_device=$3  # 0 if not passed in

# Check input
if [[ $action != 'down' && $action != 'up' ]]; then
    echo "Wrong action, must be down or up"
    exit 1
fi
if [[ $imu_type != 'all' && $imu_type != 'none' ]]; then
    echo "Wrong imu_type, must be all or none"
    exit 1
fi
if [[ -z $cuda_device ]]; then
    cuda_device=0
fi

# Main work
echo "Running with scroll ${action} and ${imu_type}"
if [[ $imu_type == 'all' ]]; then 
    config_file="configs/humi/main/humi_scroll50${action}_imu100all_epoch500_enroll3_b128.yaml"
elif [[ $imu_type == 'none' ]]; then
    config_file="configs/humi/main/humi_scroll50${action}_epoch500_enroll3_b128.yaml"
fi

cmd="CUDA_VISIBLE_DEVICES=${cuda_device} python test.py --dataname humi -c ${config_file} --metric basic det pca"
echo $cmd
eval $cmd