#!/bin/bash

# 변수 설정
num_cities=50
constraint_type="path"
now=$(date +"%F_%T")
run_name="tsp${num_cities}_2opt_${now}"

# 시스템의 CPU 코어 수 확인
num_cores=$(nproc)
# 총 데이터 개수 및 기본 interval 설정
total_samples=1280

# 각 코어당 처리할 샘플 수 계산 (최소 1개의 코어당 한 interval을 처리하도록 함)
interval=$((total_samples / num_cores))
if [ $interval -lt 1 ]; then
    interval=1
fi

# 로그 디렉토리가 존재하지 않으면 생성
log_dir="/mnt/home/zuwang/workspace/diffusion_rl_tsp/logs/${constraint_type}/tsp${num_cities}_2opt_${now}"
mkdir -p ${log_dir}


# nohup python 2opt.py --run_name ${run_name} --num_cities ${num_cities} --constraint_type ${constraint_type} > "/mnt/home/zuwang/workspace/diffusion_rl_tsp/logs/${constraint_type}/${run_name}.log" 2>&1 &
for ((i=0; i<1280; i+=interval))
do
    start_idx=$i
    end_idx=$((i+interval))
    run_name="tsp${num_cities}_2opt_${start_idx}_${end_idx}_${now}"

    # nohup 명령어를 사용하여 백그라운드에서 Python 스크립트를 실행합니다.
    nohup python 2opt.py --run_name ${run_name} --num_cities ${num_cities} --constraint_type ${constraint_type} --start_idx ${start_idx} --end_idx ${end_idx} > "${log_dir}/${run_name}.log" 2>&1 &
done