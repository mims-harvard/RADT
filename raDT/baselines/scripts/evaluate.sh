#!/bin/bash
# Evaluate a baseline checkpoint (path relative to CHECKPOINT_DIR), e.g.
#   bash scripts/evaluate.sh FetchReachObstacle baselines/FetchReachObstacle/<run_name>/<run_name>-Epoch<epoch>.pt --bsa_box_size 0.1
#   bash scripts/evaluate.sh PointMazeObstacle baselines/PointMazeObstacle/<run_name>/<run_name>-Epoch<epoch>.pt --maze U_MAZE --num_avoid 3
set -e
cd "$(dirname "$0")/.."
env=$1; model_path=$2; shift 2
python3 evaluate.py --env $env --model_path $model_path --num_ep 60 "$@"
