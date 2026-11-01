import os
import argparse
import numpy as np
import torch
import gymnasium as gym

from raDT.baselines.envs import register_envs, maps_obstacle
from raDT.baselines.mpi_utils.normalizer import normalizer
from raDT.constants import CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='FetchReachObstacle', choices=['FetchReachObstacle', 'PointMazeObstacle'])
parser.add_argument('--model_path', type=str, required=True, help='model checkpoint (.pt), relative to CHECKPOINT_DIR in raDT/constants.py')
parser.add_argument('--num_ep', type=int, default=60, help='number of evaluation episodes')
# FetchReachObstacle
parser.add_argument('--bsa_box_size', type=float, default=None, help='half of the avoid-box width')
# PointMazeObstacle
parser.add_argument('--maze', type=str, default='U_MAZE', choices=['U_MAZE', 'A_MAZE', 'H_MAZE'])
parser.add_argument('--num_avoid', type=int, default=1, help='number of avoid regions in the maze')
parser.add_argument('--num_avoid_training', type=int, default=None, help='number of avoid regions the model was trained with (defaults to --num_avoid)')
parser.add_argument('--max_episode_steps', type=int, default=None, help='steps per episode (default: 50 for FetchReachObstacle, 300 for U_MAZE, 800 for A_MAZE and H_MAZE)')
args = parser.parse_args()

register_envs()
if args.env == 'FetchReachObstacle':
    max_episode_steps = args.max_episode_steps or 50
    env = gym.make('FetchReachObstacle-v0', bsa_box_size=args.bsa_box_size, max_episode_steps=max_episode_steps)
else:
    max_episode_steps = args.max_episode_steps or (300 if args.maze == 'U_MAZE' else 800)
    env = gym.make('PointMazeObstacle-v0', maze_map=getattr(maps_obstacle, args.maze),
                   num_avoid=args.num_avoid, max_episode_steps=max_episode_steps)

# extra avoid-region centroids (2 dims each) that a maze model trained with fewer regions does not see
num_avoid_training = args.num_avoid_training or args.num_avoid
n_extra_dims = 2 * (args.num_avoid - num_avoid_training) if args.env == 'PointMazeObstacle' else 0

o_norm_mean, o_norm_std, g_norm_mean, g_norm_std, actor_network = torch.load(os.path.join(CHECKPOINT_DIR, args.model_path), map_location=torch.device("cpu"))
obs, info = env.reset()
o_norm = normalizer(size=obs['observation'].shape[0] - n_extra_dims, default_clip_range=5)
o_norm.mean, o_norm.std = o_norm_mean, o_norm_std
g_norm = normalizer(size=obs['desired_goal'].shape[0], default_clip_range=5)
g_norm.mean, g_norm.std = g_norm_mean, g_norm_std

def preproc_inputs(obs, g):
    inputs = np.concatenate([o_norm.normalize(obs), g_norm.normalize(g)])
    return torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)

ep_cost, ep_len = 0, 0
normalized_costs, successes = [], []
while len(successes) < args.num_ep:
    observation = obs["observation"]
    if n_extra_dims:
        observation = observation[:len(observation) - n_extra_dims]
    with torch.no_grad():
        act = actor_network(preproc_inputs(observation, obs["desired_goal"])).numpy().flatten()
    obs, reward, cost, done, info = env.step(act)
    ep_cost += cost
    ep_len += 1
    if done or info['is_success']: # stop early if goal reached
        successes.append(info['is_success'])
        normalized_costs.append(ep_cost / ep_len)
        ep_cost, ep_len = 0, 0
        obs, info = env.reset()
env.close()

print("Mean Normalized Cost: ", np.mean(normalized_costs))
print("Success Rate: ", np.mean(successes))
