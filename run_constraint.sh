#!/bin/bash

# Configuration variables
interval=20
num_cities=200
num_init_sample=3
num_epochs=3
num_inner_epochs=3
now=$(date +"%F_%T")
run_name="tsp${num_cities}_epoch${num_epochs}_${num_inner_epochs}_${now}"
constraint_type="cluster"

# Create log directory
mkdir -p "./logs/${constraint_type}/${run_name}"

# Manually define available devices
available_devices=(0 1 2 3)  # Specify the GPU IDs you want to use
num_devices=${#available_devices[@]}

# Train models in batches
for (( start_idx=0; start_idx+interval<=1280; start_idx+=interval )); do
    end_idx=$((start_idx + interval))
    
    # Set CUDA device
    device_number=${available_devices[$(( (start_idx / interval) % num_devices ))]}
    CUDA_VISIBLE_DEVICES=$device_number nohup python train_constraint.py --run_name $run_name --start_idx $start_idx --end_idx $end_idx --num_cities $num_cities --num_epochs $num_epochs --num_inner_epochs $num_inner_epochs \
    --constraint_type $constraint_type \
        > "./logs/${constraint_type}/${run_name}/from${start_idx}_to${end_idx}.log" 2>&1 &
done
