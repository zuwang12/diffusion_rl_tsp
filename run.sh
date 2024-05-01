#!/bin/bash

interval=20
num_cities=20
run_name='pp_init_test'
mkdir -p ./logs/${run_name}

device_number_tmp=0
for (( start_idx=0; start_idx+interval<=1280; start_idx+=interval ))
do
    end_idx=$((start_idx + interval))
    device_number=$((device_number_tmp%4))
    CUDA_VISIBLE_DEVICES=$device_number nohup python train_one.py --sample_idx_min $start_idx --sample_idx_max $end_idx --num_cities $num_cities --run_name $run_name > ./logs/${run_name}/${num_cities}cities_from${start_idx}_to${end_idx}.log 2>&1 &
    device_number_tmp=$((device_number_tmp+1))
done