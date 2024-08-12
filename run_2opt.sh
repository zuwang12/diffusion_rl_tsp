#!/bin/bash

# 변수 설정
num_cities=200
constraint_type="cluster"
now=$(date +"%F_%T")
run_name="tsp${num_cities}_2opt_${now}"

# 로그 디렉토리가 존재하지 않으면 생성
log_dir="/mnt/home/zuwang/workspace/diffusion_rl_tsp/logs/${constraint_type}"
mkdir -p ${log_dir}

# nohup 명령어를 사용하여 백그라운드에서 Python 스크립트를 실행합니다.
nohup python 2opt.py --run_name ${run_name} --num_cities ${num_cities} --constraint_type ${constraint_type} > "/mnt/home/zuwang/workspace/diffusion_rl_tsp/logs/${constraint_type}/${run_name}.log" 2>&1 &
