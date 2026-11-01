import os
import pickle
import numpy as np

from raDT.constants import BASELINE_DATA_DIR, RADT_DATASET_DIR


def load_random_policy_data(env_name, buffer_name):
    # random-policy dataset written by baselines/generate_data.py, converted to a list of RADT trajectories
    with open(os.path.join(BASELINE_DATA_DIR, 'random', env_name, f'{buffer_name}.pkl'), 'rb') as f:
        data = pickle.load(f)

    trajectories = []
    for i in range(data['o'].shape[0]):
        traj = {}
        traj['observations'] = data['o'][i, :-1, :]
        traj['next_observations'] = data['o'][i, 1:, :]
        traj['actions'] = data['u'][i, :, :]
        traj['timesteps'] = traj['observations'].shape[0]
        traj['rewards'] = data['r'][i, :, :].flatten()
        traj['costs'] = data['c'][i, :, :].flatten()
        traj['terminals'] = np.array([False] * (traj['timesteps'] - 1) + [True])
        traj['goal_pos'] = list(data['g'][i, 0])
        trajectories.append(traj)
    return trajectories


def avoids_all(positions, boxes):
    # True if no position lies inside any box ([lower bounds, upper bounds])
    d = positions.shape[1]
    for box in boxes:
        if np.any(np.all(positions >= box[:d], axis=1) & np.all(positions <= box[d:], axis=1)):
            return False
    return True


def save_dataset(trajectories, filename):
    os.makedirs(RADT_DATASET_DIR, exist_ok=True)
    path = os.path.join(RADT_DATASET_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(trajectories, f)
    print('saved', path)
