#!/bin/bash
# Random-policy datasets (2e6 steps each), written to BASELINE_DATA_DIR/random/<env>/.
# Used to train the baselines and, after relabeling, RADT.
set -e
cd "$(dirname "$0")/.."

# FetchReachObstacle
python3 generate_data.py --env FetchReachObstacle --num_timesteps 2e6
# cost labels for the larger avoid boxes, used to retrain the baselines for each box size
for width in 0.18 0.20 0.22 0.24; do
    python3 recompute_cost.py --box_width $width
done

# MazeObstacle (num_avoid 1 is also the source data for RADT)
for num_avoid in 1 3 4 5 6 7; do
    python3 generate_data.py --env PointMazeObstacle --maze U_MAZE --num_avoid $num_avoid --num_timesteps 2e6
done
for maze in A_MAZE H_MAZE; do
    for num_avoid in 1 3 5 7; do
        python3 generate_data.py --env PointMazeObstacle --maze $maze --num_avoid $num_avoid --num_timesteps 2e6
    done
done
