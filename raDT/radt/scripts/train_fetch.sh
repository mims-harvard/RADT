#!/bin/bash
# Table 1: RADT on FetchReachObstacle, trained with avoid boxes up to width 0.16 (3 seeds).
# Checkpoints are saved and evaluated every 500 iterations; see the README for checkpoint selection.
set -e
cd "$(dirname "$0")/.."

for seed in 0 1 2; do
    python3 train.py \
        --exp-name radt-fetch \
        --instance_prefix seed$seed \
        --seed $seed \
        --env reach_obstacle \
        --dataset_path reach_obstacle_relabeled.pkl \
        --n_head 4 --n_layer 4 --embed_dim 256 \
        --adelta 2 --alpha1 1 --alpha2 1.1 \
        --max_ep_len 50 \
        --batch_size 128 --learning_rate 1e-4 \
        --scheduler cosinewarmuprestarts --T_0 10000 --warmup_steps 5000 \
        --max_iters 50000 --eval_interval 500 --save_interval 500 \
        --num_eval_episodes 60 \
        --bsa_box_size 0.08
done
