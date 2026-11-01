#!/bin/bash
# Results 3: RADT on Cardiogenesis (run cardiogenesis_data.ipynb first), then evaluate with cardiogenesis_eval.ipynb.
set -e
cd "$(dirname "$0")/.."

python3 train.py \
    --exp-name radt-cardio \
    --instance_prefix run \
    --env cardiogenesis \
    --dataset_path cardiogenesis_relabeled.pkl \
    --n_head 6 --n_layer 6 --embed_dim 384 \
    --adelta 1 --alpha1 1 --alpha2 1.1 \
    --max_ep_len 30 \
    --batch_size 128 --learning_rate 1e-4 \
    --scheduler lambdalr --warmup_steps 10000 \
    --max_iters 10000 --eval_interval 100 --save_interval 100 \
    --num_eval_episodes 60 \
    --fixed_interval 10
