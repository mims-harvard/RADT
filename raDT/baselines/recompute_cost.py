"""
Recompute the cost labels of a FetchReachObstacle random-policy dataset for a different
avoid-box size, so that the baselines can be retrained for each box size.

The cost uses the same definition as FetchReachObstacleEnv.compute_cost with bsa_box_size:
the step into o[t + 1] costs 1 iff the gripper is inside the box of half-width bsa_box_size
around the obstacle and the goal has not been reached. Trajectories are unchanged.
"""
import argparse
import os
import pickle
import numpy as np

from raDT.constants import BASELINE_DATA_DIR

DISTANCE_THRESHOLD = 0.05  # FetchReachObstacleEnv distance_threshold

parser = argparse.ArgumentParser()
parser.add_argument('--box_width', type=float, required=True, help='full width of the avoid box, e.g. 0.20')
parser.add_argument('--buffer', type=str, default=os.path.join(BASELINE_DATA_DIR, 'random', 'FetchReachObstacle', 'buffer_massless.pkl'))
args = parser.parse_args()

with open(args.buffer, 'rb') as f:
    d = pickle.load(f)
o, ag, g, c = d['o'], d['ag'], d['g'], d['c']
T = c.shape[1]

next_o = o[:, 1:T + 1]
inside = (np.abs(next_o[..., 0:3] - next_o[..., 10:13]) < args.box_width / 2).all(axis=-1)
not_reached = np.linalg.norm(ag[:, 1:T + 1] - g[:, :T], axis=-1) > DISTANCE_THRESHOLD
d['c'] = (inside & not_reached).astype(c.dtype)[..., np.newaxis]

out_path = args.buffer.replace('.pkl', f'_box{args.box_width:.2f}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(d, f)
print(f'{out_path}: mean cost {c.mean():.4f} -> {d["c"].mean():.4f}')
