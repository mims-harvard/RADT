#!/bin/bash
# Table 1 baselines: retrained for each avoid-box width (3 seeds each).
# The model is evaluated and saved after every epoch; see the README for checkpoint selection.
set -e
cd "$(dirname "$0")/.."

for method in AMlag rbsl wgcsl; do
    for width in 0.16 0.18 0.20 0.22 0.24; do
        half_width=$(python3 -c "print($width / 2)")
        suffix=$([ "$width" = "0.16" ] && echo "" || echo "_box$width")
        python3 train.py \
            --env FetchReachObstacle \
            --method $method \
            --bsa_box_size $half_width \
            --buffer_suffix "$suffix" \
            --n-epochs 500 \
            --n-test-rollouts 60 \
            --n_seeds 3
    done
done
