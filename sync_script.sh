#!/bin/bash

# Define the remote server and port
REMOTE_SERVER="zuwang@14.32.115.197"
REMOTE_PORT="54329"

# Define the local and remote directories
LOCAL_RESULTS_DIR="./Results"
REMOTE_RESULTS_DIR="/home/zuwang/workspace/diffusion_rl_tsp/Results"

LOCAL_LOGS_DIR="./logs"
REMOTE_LOGS_DIR="/home/zuwang/workspace/diffusion_rl_tsp/logs"

# Rsync commands for Results directories
rsync -avz -e "ssh -p $REMOTE_PORT" $LOCAL_RESULTS_DIR/basic/* $REMOTE_SERVER:$REMOTE_RESULTS_DIR/basic/
rsync -avz -e "ssh -p $REMOTE_PORT" $LOCAL_RESULTS_DIR/box/* $REMOTE_SERVER:$REMOTE_RESULTS_DIR/box/
rsync -avz -e "ssh -p $REMOTE_PORT" $LOCAL_RESULTS_DIR/path/* $REMOTE_SERVER:$REMOTE_RESULTS_DIR/path/
rsync -avz -e "ssh -p $REMOTE_PORT" $LOCAL_RESULTS_DIR/cluster/* $REMOTE_SERVER:$REMOTE_RESULTS_DIR/cluster/

# Rsync commands for Logs directories
rsync -avz -e "ssh -p $REMOTE_PORT" $LOCAL_LOGS_DIR/basic/* $REMOTE_SERVER:$REMOTE_LOGS_DIR/basic/
rsync -avz -e "ssh -p $REMOTE_PORT" $LOCAL_LOGS_DIR/box/* $REMOTE_SERVER:$REMOTE_LOGS_DIR/box/
rsync -avz -e "ssh -p $REMOTE_PORT" $LOCAL_LOGS_DIR/path/* $REMOTE_SERVER:$REMOTE_LOGS_DIR/path/
rsync -avz -e "ssh -p $REMOTE_PORT" $LOCAL_LOGS_DIR/cluster/* $REMOTE_SERVER:$REMOTE_LOGS_DIR/cluster/
