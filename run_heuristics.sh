#!/bin/bash

# 변수 설정
num_cities=100
constraint_type="path"
now=$(date +"%F_%T")
run_name="tsp${num_cities}_2opt_${now}"

# nohup 명령어를 사용하여 백그라운드에서 Python 스크립트를 실행합니다.
nohup python heuristics.py --run_name ${run_name} --num_cities ${num_cities} --constraint_type ${constraint_type} > "/mnt/home/zuwang/workspace/diffusion_rl_tsp/logs/${constraint_type}/${run_name}.log" 2>&1 &
