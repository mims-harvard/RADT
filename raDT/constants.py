import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# User settings: this is the only file you need to edit.
# All defaults point inside the repository; change them to wherever you want
# large files (datasets, checkpoints, logs) to live on your system.
# ---------------------------------------------------------------------------

# Random-policy datasets written by baselines/generate_data.py.
# Read by the baselines for training and by radt/relabel_*.py for relabeling.
BASELINE_DATA_DIR = os.path.join(REPO_ROOT, "raDT", "baselines", "offline_data")

# Relabeled datasets used to train RADT (written by radt/relabel_*.py and
# radt/cardiogenesis_data.ipynb).
RADT_DATASET_DIR = os.path.join(REPO_ROOT, "raDT", "radt", "dataset")

# Model checkpoints for RADT and the baselines.
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")

# Weights & Biases logging.
WANDB_DIR = os.path.join(REPO_ROOT, "wandb")
WANDB_ENTITY = None  # your wandb username or team; None uses your default entity
WANDB_PROJECT_RADT = "radt"
WANDB_PROJECT_BASELINES = "radt-baselines"

# ---------------------------------------------------------------------------
# Paths inside the repository (no need to edit).
# ---------------------------------------------------------------------------
RADT_PATH = os.path.join(REPO_ROOT, "raDT", "radt")
BASELINES_PATH = os.path.join(REPO_ROOT, "raDT", "baselines")
