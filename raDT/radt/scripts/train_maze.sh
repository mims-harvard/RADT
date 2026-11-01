#!/bin/bash
# Table 2: RADT on the MazeObstacle environments (3 seeds each).
# UMaze is trained with 1 avoid region (in-distribution row) and with 3 avoid regions (evaluated on 3-7);
# AMaze and HMaze are trained with 3 avoid regions.
# Checkpoints are saved and evaluated every 500 iterations; see the README for checkpoint selection.
set -e
cd "$(dirname "$0")/.."

train() {  # maze, max_ep_len, num_avoid, seed
    python3 train.py \
        --exp-name radt-maze \
        --instance_prefix $1-na$3-seed$4 \
        --seed $4 \
        --env pointmaze_obstacle \
        --dataset_path pointmaze_obstacle_$(echo $1 | tr 'A-Z' 'a-z' | tr -d '_')$2_numavoid$3_relabeled.pkl \
        --maze $1 --num_avoid $3 \
        --n_head 6 --n_layer 6 --embed_dim 384 \
        --adelta 1 --alpha1 1 --alpha2 1.1 \
        --max_ep_len $2 \
        --batch_size 32 --learning_rate 1e-4 \
        --scheduler lambdalr --warmup_steps 10000 \
        --max_iters 50000 --eval_interval 500 --save_interval 500 \
        --num_eval_episodes 60
}

for seed in 0 1 2; do
    train U_MAZE 300 1 $seed
    train U_MAZE 300 3 $seed
    train A_MAZE 800 3 $seed
    train H_MAZE 800 3 $seed
done
