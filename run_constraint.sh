#!/bin/bash

# Configuration variables
interval=20
num_cities=50
num_init_sample=3
num_epochs=1
num_inner_epochs=1
now=$(date +"%F_%T")
run_name="tsp${num_cities}_epoch${num_epochs}_${num_inner_epochs}_${now}"

# Create log directory
mkdir -p "./logs/${run_name}"

# Detect the number of available GPUs
num_gpus=$(nvidia-smi -L | wc -l)  # Automatically detect the number of GPUs available
device_number=0

# Train models in batches
for (( start_idx=0; start_idx+interval<=1280; start_idx+=interval )); do
    end_idx=$((start_idx + interval))
    
    # Set CUDA device
    CUDA_VISIBLE_DEVICES=$device_number nohup python train_constraint_path.py --run_name $run_name --start_idx $start_idx --end_idx $end_idx --num_cities $num_cities \
        > "./logs/${run_name}/from${start_idx}_to${end_idx}.log" 2>&1 &

    # Rotate through available CUDA devices
    device_number=$(( (device_number + 1) % num_gpus ))
done
