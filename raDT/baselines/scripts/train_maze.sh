#!/bin/bash
# Table 2 baselines: retrained for each number of avoid regions (3 seeds each).
# The model is evaluated and saved after every epoch; see the README for checkpoint selection.
set -e
cd "$(dirname "$0")/.."

for method in AMlag rbsl wgcsl; do
    for num_avoid in 1 3 4 5 6 7; do
        python3 train.py --env PointMazeObstacle --method $method --maze U_MAZE --num_avoid $num_avoid \
            --n-epochs 500 --n-test-rollouts 60 --n_seeds 3
    done
done

# AMazeObstacle and HMazeObstacle: AM-Lag only
for maze in A_MAZE H_MAZE; do
    for num_avoid in 3 5 7; do
        python3 train.py --env PointMazeObstacle --method AMlag --maze $maze --num_avoid $num_avoid \
            --n-epochs 500 --n-test-rollouts 60 --n_seeds 3
    done
done
