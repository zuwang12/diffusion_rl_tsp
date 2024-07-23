#!/bin/bash

# Configuration variables
interval=20
num_cities=50
num_init_sample=10
num_epochs=20
num_inner_epochs=50
now=$(date +"%F_%T")
run_name="tsp${num_cities}_epoch${num_epochs}_${num_inner_epochs}_sample${num_init_sample}"

# Create log directory
mkdir -p "./logs/${run_name}_${now}"

# Train models in batches
device_number=0
for (( start_idx=0; start_idx+interval<=1280; start_idx+=interval )); do
    end_idx=$((start_idx + interval))
    
    # Set CUDA device
    CUDA_VISIBLE_DEVICES=$device_number nohup python train_one.py \
        --sample_idx_min $start_idx \
        --sample_idx_max $end_idx \
        --num_cities $num_cities \
        --num_init_sample $num_init_sample \
        --num_epochs $num_epochs \
        --num_inner_epochs $num_inner_epochs \
        --run_name $run_name \
        --now $now \
        > "./logs/${run_name}_${now}/${num_cities}cities_from${start_idx}_to${end_idx}.log" 2>&1 &

    # Rotate through CUDA devices 0-3
    device_number=$(( (device_number + 1) % 4 ))
done
