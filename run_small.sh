#!/bin/bash

# Configuration variables
interval=32
num_cities=500
img_size=128
max_iter=10000
num_init_sample=1
num_epochs=1
num_inner_epochs=1
constraint_type="basic"

now=$(date +"%F_%T")
run_name="tsp${num_cities}_epoch${num_epochs}_${num_inner_epochs}_${now}"

# Create log directory
log_dir="./logs/${constraint_type}/${run_name}"
mkdir -p "$log_dir"

# Define available GPU devices manually
available_devices=(0 1 2 3)
num_devices=${#available_devices[@]}

data_size=128

# Train models in batches
for (( start_idx=0; start_idx<data_size; start_idx+=interval )); do
    end_idx=$((start_idx + interval))
    if (( end_idx > data_size )); then
        end_idx=$data_size
    fi
    
    # Assign CUDA device based on batch index
    device_number=${available_devices[$(( (start_idx / interval) % num_devices ))]}

    # Run training with nohup and save logs
    CUDA_VISIBLE_DEVICES=$device_number nohup python main.py \
        --run_name "$run_name" \
        --start_idx "$start_idx" \
        --end_idx "$end_idx" \
        --num_cities "$num_cities" \
        --img_size "$img_size" \
        --max_iter "$max_iter" \
        --num_epochs "$num_epochs" \
        --num_inner_epochs "$num_inner_epochs" \
        --num_init_sample "$num_init_sample" \
        --constraint_type "$constraint_type" \
        > "$log_dir/from${start_idx}_to${end_idx}.log" 2>&1 &
done
