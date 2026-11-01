#!/bin/bash
# Zero-shot evaluation of trained RADT checkpoints (paths relative to CHECKPOINT_DIR), e.g.
#   bash scripts/evaluate.sh fetch radt/<run_name>/prompt_model_reach_obstacle_iter_<iter>
#   bash scripts/evaluate.sh umaze1 radt/<run_name>/prompt_model_pointmaze_obstacle_iter_<iter>
#   bash scripts/evaluate.sh umaze3 radt/<run_name>/prompt_model_pointmaze_obstacle_iter_<iter>
#   bash scripts/evaluate.sh amaze radt/<run_name>/prompt_model_pointmaze_obstacle_iter_<iter>
#   bash scripts/evaluate.sh hmaze radt/<run_name>/prompt_model_pointmaze_obstacle_iter_<iter>
set -e
cd "$(dirname "$0")/.."
setting=$1; checkpoint=$2

maze_eval() {  # maze, max_ep_len, num_avoid values
    for num_avoid in "${@:3}"; do
        echo "### $1, $num_avoid avoid regions"
        python3 train.py --evaluation --load-path $checkpoint --env pointmaze_obstacle \
            --maze $1 --num_avoid $num_avoid --max_ep_len $2 \
            --n_head 6 --n_layer 6 --embed_dim 384 --adelta 1 --num_eval_episodes 200 --no_wandb
    done
}

case $setting in
    fetch)
        # avoid-box widths 0.16 (in-distribution) to 0.24
        for half_width in 0.08 0.09 0.10 0.11 0.12; do
            echo "### box width $(python3 -c "print(2 * $half_width)")"
            python3 train.py --evaluation --load-path $checkpoint --env reach_obstacle \
                --bsa_box_size $half_width --max_ep_len 50 \
                --n_head 4 --n_layer 4 --embed_dim 256 --adelta 2 --num_eval_episodes 60 --no_wandb
        done ;;
    umaze1) maze_eval U_MAZE 300 1 ;;
    umaze3) maze_eval U_MAZE 300 3 4 5 6 7 ;;
    amaze) maze_eval A_MAZE 800 3 5 7 ;;
    hmaze) maze_eval H_MAZE 800 3 5 7 ;;
    *) echo "unknown setting $setting"; exit 1 ;;
esac
