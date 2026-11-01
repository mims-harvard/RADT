import os
import argparse
import pickle
import numpy as np
import gymnasium as gym

from raDT.baselines.envs import register_envs, maps_obstacle
from raDT.constants import BASELINE_DATA_DIR

MAZE_NAMES = {'U_MAZE': 'umaze', 'A_MAZE': 'amaze', 'H_MAZE': 'hmaze'}

parser = argparse.ArgumentParser(description='collect random-policy trajectories')
parser.add_argument('--env', type=str, default='FetchReachObstacle', choices=['FetchReachObstacle', 'PointMazeObstacle'])
parser.add_argument('--num_timesteps', type=lambda x: int(float(x)), default=int(2e6), help='number of random actions to take')
parser.add_argument('--seed', type=int, default=None)
# PointMazeObstacle
parser.add_argument('--maze', type=str, default='U_MAZE', choices=list(MAZE_NAMES))
parser.add_argument('--num_avoid', type=int, default=1, help='number of avoid regions in the maze')
parser.add_argument('--max_episode_steps', type=int, default=None, help='steps per episode (default: 300 for U_MAZE, 800 for A_MAZE and H_MAZE)')
args = parser.parse_args()

register_envs()
if args.env == 'FetchReachObstacle':
    env = gym.make('FetchReachObstacle-v0')
    buffer_name = 'buffer_massless'
else:
    max_episode_steps = args.max_episode_steps or (300 if args.maze == 'U_MAZE' else 800)
    env = gym.make('PointMazeObstacle-v0', maze_map=getattr(maps_obstacle, args.maze),
                   num_avoid=args.num_avoid, max_episode_steps=max_episode_steps)
    buffer_name = f'buffer_numavoid{args.num_avoid}_{MAZE_NAMES[args.maze]}{max_episode_steps}'

out_dir = os.path.join(BASELINE_DATA_DIR, 'random', args.env)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'{buffer_name}.pkl')

if args.seed is not None:
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
obs, info = env.reset(seed=args.seed)

o_ll, ag_ll, g_ll, u_ll, r_ll, c_ll = [], [], [], [], [], []
o_list, ag_list, g_list, u_list, r_list, c_list = [], [], [], [], [], []
o_list.append(obs["observation"])
ag_list.append(obs["achieved_goal"])
g_list.append(obs["desired_goal"])

for t in range(args.num_timesteps):
    if t % 100000 == 0:
        print(t)

    act = env.action_space.sample()
    u_list.append(act)
    obs, reward, cost, done, info = env.step(act)
    r_list.append(reward)
    c_list.append(cost)
    o_list.append(obs["observation"])
    ag_list.append(obs["achieved_goal"])
    g_list.append(obs["desired_goal"])

    if done:
        o_ll.append(np.stack(o_list))
        ag_ll.append(np.stack(ag_list))
        g_ll.append(np.stack(g_list[:-1]))
        u_ll.append(np.stack(u_list))
        r_ll.append(np.stack(r_list))
        c_ll.append(np.stack(c_list))

        o_list, ag_list, g_list, u_list, r_list, c_list = [], [], [], [], [], []

        obs, info = env.reset()
        o_list.append(obs["observation"])
        ag_list.append(obs["achieved_goal"])
        g_list.append(obs["desired_goal"])

# the cost c[:, t] belongs to the transition o[:, t] -> o[:, t + 1]
d = {'o': np.stack(o_ll)[:,:-1,:],
     'ag': np.stack(ag_ll)[:,:-1,:],
     'g': np.stack(g_ll)[:,:-1,:],
     'u': np.stack(u_ll)[:,:-1,:],
     'r': np.stack(r_ll)[:,:-1,np.newaxis],
     'c': np.stack(c_ll)[:,:-1,np.newaxis]}

with open(out_path, 'wb') as handle:
    pickle.dump(d, handle)
print('saved', out_path)
