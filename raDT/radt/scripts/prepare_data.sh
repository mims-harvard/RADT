#!/bin/bash
# Hindsight avoid-region relabeling of the random-policy data (run baselines/scripts/generate_data.sh first).
# Datasets are written to RADT_DATASET_DIR.
set -e
cd "$(dirname "$0")/.."

python3 relabel_fetch.py --max_half_width 0.08 --output reach_obstacle_relabeled.pkl

python3 relabel_maze.py --maze U_MAZE --num_avoid 1
for maze in U_MAZE A_MAZE H_MAZE; do
    python3 relabel_maze.py --maze $maze --num_avoid 3
done

# Cardiogenesis: run cardiogenesis_data.ipynb
