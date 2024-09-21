#!/bin/bash

# 변수 설정
num_cities=100
max_iter=10
constraint_type="path"
now=$(date +"%F_%T")  # %F는 YYYY-MM-DD, %T는 HH:MM:SS 형식
run_name="tsp${num_cities}_2opt_max${max_iter}_${now}"

# 시스템의 CPU 코어 수 확인
num_cores=$(nproc)
# 총 데이터 개수 및 기본 interval 설정
total_samples=1280

# 각 코어당 처리할 샘플 수 계산 (최소 1개의 코어당 한 interval을 처리하도록 함)
interval=$((total_samples / num_cores))
if [ $interval -lt 1 ]; then
    interval=1
fi
interval=20

# 로그 디렉토리가 존재하지 않으면 생성
log_dir="/mnt/home/zuwang/workspace/diffusion_rl_tsp/logs/${constraint_type}/${run_name}"
mkdir -p "${log_dir}"

# 반복적으로 Python 스크립트를 백그라운드에서 실행
for ((i=0; i<1280; i+=interval)); do
    start_idx=$i
    end_idx=$((i+interval))
    
    # end_idx가 total_samples를 초과하지 않도록 제한
    if [ $end_idx -gt $total_samples ]; then
        end_idx=$total_samples
    fi

    # 백그라운드에서 Python 스크립트 실행
    nohup python 2opt.py --run_name "${run_name}" --num_cities ${num_cities} --max_iter ${max_iter} --constraint_type ${constraint_type} --start_idx ${start_idx} --end_idx ${end_idx} > "${log_dir}/from${start_idx}_to${end_idx}.log" 2>&1 &
done
