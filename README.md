# Reach-Avoid Decision Transformer (RADT)

This repository is the official implementation of [Hindsight Relabeling is All You Need for Reach-Avoid Learning (RADT)](https://github.com/mims-harvard/RADT/tree/master) (NeurIPS 2026).

The code for training and evaluating the baselines is built on top of the [RbSL repository](https://github.com/Sunlighted/RbSL.git), and the code for training and evaluating RADT is built on top of the [MGPO repository](https://github.com/PKU-RL/MGPO.git).

## Repository structure

```
raDT/
  constants.py                  # paths and logging settings: the only file you need to edit
  radt/                         # RADT
    train.py                    # training and evaluation
    relabel_fetch.py            # hindsight goal + avoid-region relabeling, FetchReachObstacle
    relabel_maze.py             # hindsight goal + avoid-region relabeling, MazeObstacle
    cardiogenesis_data.ipynb    # data generation + relabeling, Cardiogenesis
    cardiogenesis_eval.ipynb    # avoid-state case study, Cardiogenesis
    prompt_dt/                  # model, trainer, batching and evaluation
    envs/                       # environment interfaces (FetchReach, PointMaze, Cardiogenesis)
    scripts/                    # commands for the paper's experiments
  baselines/                    # AM-Lag, RbSL, WGCSL
    generate_data.py            # random-policy data collection (used by the baselines and RADT)
    recompute_cost.py           # cost labels for other avoid-box sizes, FetchReachObstacle
    train.py                    # training (evaluates and saves a checkpoint every epoch)
    evaluate.py                 # evaluation of a saved checkpoint
    envs/                       # FetchReachObstacle and PointMazeObstacle environments
    rl_modules/, her_modules/, mpi_utils/
    scripts/                    # commands for the paper's experiments
```

## Setup

### Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate radt
```

The following packages must be installed manually:

```bash
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install 'git+https://github.com/hklarner/pyboolnet'
```

Install a CUDA-compatible PyTorch for your system by following the [PyTorch instructions](https://pytorch.org/get-started/locally/).

Finally, install this repository as a Python package (`raDT`), from the repository root. This is required for the import statements to work:

```bash
pip install -e .
```

`FetchReachObstacle` uses `mujoco-py`, which needs the [MuJoCo 2.1 binaries](https://github.com/openai/mujoco-py#install-mujoco) in `~/.mujoco/mujoco210`.

### Configure paths

**Configure system paths in `raDT/constants.py`.** It sets where datasets, checkpoints and Weights & Biases logs are written (by default, folders inside this repository) and your Weights & Biases entity and project names. All scripts read these settings; dataset and checkpoint arguments on the command line are relative to these folders.

Training logs to Weights & Biases by default; pass `--no_wandb` to `train.py` (RADT and baselines) to turn it off.

## Pipeline

All commands below are run from the repository root with the `radt` environment active. Each script runs every configuration of an experiment one after another; to run them in parallel, copy individual commands out of the scripts or submit the scripts to your job scheduler. A GPU is recommended for training.

### 1. Data prep

```bash
# random-policy data for all environments, and cost labels for the larger FetchReach avoid boxes
bash raDT/baselines/scripts/generate_data.sh
# hindsight avoid-region relabeling for RADT
bash raDT/radt/scripts/prepare_data.sh
```

For Cardiogenesis, run `raDT/radt/cardiogenesis_data.ipynb` (from `raDT/radt`). It generates the random-policy data and relabels it in one go.

### 2. Training and evaluation

| Result | RADT | Baselines |
|---|---|---|
| FetchReachObstacle, avoid-box widths 0.16 (in-distribution) to 0.24 | `bash raDT/radt/scripts/train_fetch.sh`<br>`bash raDT/radt/scripts/evaluate.sh fetch <checkpoint>` | `bash raDT/baselines/scripts/train_fetch.sh` (retrained for each width) |
| UMazeObstacle, 1 avoid region (in-distribution) | `bash raDT/radt/scripts/train_maze.sh`<br>`bash raDT/radt/scripts/evaluate.sh umaze1 <checkpoint>` | `bash raDT/baselines/scripts/train_maze.sh` (retrained for each number of avoid regions) |
| UMazeObstacle, 3 avoid regions (in-distribution) and 4-7 (zero-shot) | `bash raDT/radt/scripts/evaluate.sh umaze3 <checkpoint>` | |
| AMazeObstacle and HMazeObstacle, 3 avoid regions (in-distribution), 5 and 7 (zero-shot) | `bash raDT/radt/scripts/evaluate.sh amaze <checkpoint>`<br>`bash raDT/radt/scripts/evaluate.sh hmaze <checkpoint>` | |
| Cardiogenesis | `bash raDT/radt/scripts/train_cardio.sh`<br>then `raDT/radt/cardiogenesis_eval.ipynb` | |

`train_maze.sh` trains all RADT maze models: UMaze with 1 and with 3 avoid regions, and AMaze and HMaze with 3 avoid regions. UMaze episodes are 300 steps long and AMaze/HMaze episodes are 800 steps long. RADT models are trained with 3 seeds (`--seed 0/1/2`); the baselines with 3 seeds per run (`--n_seeds 3`).

Baseline checkpoints are evaluated with `bash raDT/baselines/scripts/evaluate.sh <env> <checkpoint> [options]`; see the examples at the top of that script.

### Notes

- For RADT, `--max_iters`, `--eval_interval` and `--save_interval` count training iterations, each consisting of `--num_steps_per_iter` (10) gradient steps. The learning-rate scheduler arguments `--warmup_steps` and `--T_0` count gradient steps.
- RADT trains with up to `--max_avoid_prompt_len` (10) avoid tokens. At evaluation time it is prompted with the environment's goal and avoid regions and with the avoid success indicator set to 1.
- In `FetchReachObstacle`, `--bsa_box_size` is half of the avoid-box width. In the maze environments, avoid regions are circles of radius 0.2 and RADT is prompted with the circumscribing boxes.
- In `Cardiogenesis`, avoid tokens are the 15-dimensional avoid states, and the goal of the case study is the first heart field (FHF) attractor `100010010101100`.
- See `python3 <script> --help` for all options of each script.

## Contact
For questions, please contact Kevin Li at <likevin@mit.edu>.


## License
The code in this package is licensed under the MIT License.

## Citation

```bibtex
@inproceedings{li2026radt,
  title     = {Hindsight Relabeling is All You Need for Reach-Avoid Learning},
  author    = {Li, Kevin and Zitnik, Marinka},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2505.19337},
  archivePrefix = {arXiv}
}
```