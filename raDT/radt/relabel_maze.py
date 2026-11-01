"""
Hindsight avoid-region relabeling for the MazeObstacle environments (Appendix A.3).

Pass 1: `num_avoid` avoid boxes per trajectory, with centroids sampled uniformly from the
accessible (non-wall) part of the maze. z = 1 if the trajectory never enters any box.
Pass 2: a copy of every trajectory, with all boxes resampled until z is the opposite of pass 1.
"""
import argparse
import copy
import math
import numpy as np

from raDT.baselines.envs import maps_obstacle
from relabel_common import load_random_policy_data, avoids_all, save_dataset

STATE_DIM = 4  # PointMaze observation without the avoid-region centroids
MAZE_NAMES = {'U_MAZE': 'umaze', 'A_MAZE': 'amaze', 'H_MAZE': 'hmaze'}


class AccessibleRegionSampler:
    # uniform samples from the interior of the maze (inside the outer walls), rejecting points in inner walls
    def __init__(self, maze_map):
        self.maze_map = maze_map
        self.n_rows, self.n_cols = len(maze_map), len(maze_map[0])
        # cell (i, j) covers x in [j - n_cols / 2, j + 1 - n_cols / 2], y in [n_rows / 2 - i - 1, n_rows / 2 - i]
        self.x_min, self.width = 1 - self.n_cols / 2, self.n_cols - 2
        self.y_min, self.height = 1 - self.n_rows / 2, self.n_rows - 2

    def in_wall(self, p):
        i = math.floor(self.n_rows / 2 - p[1])
        j = math.floor(p[0] + self.n_cols / 2)
        return self.maze_map[i][j] == 1

    def sample(self):
        p = np.array([np.random.rand() * self.width + self.x_min, np.random.rand() * self.height + self.y_min])
        while self.in_wall(p):
            p = np.array([np.random.rand() * self.width + self.x_min, np.random.rand() * self.height + self.y_min])
        return p


def sample_boxes(sampler, num_avoid, max_half_width):
    boxes = []
    for _ in range(num_avoid):
        center = sampler.sample()
        half_width = (0.75 + 0.25 * np.random.rand()) * max_half_width
        boxes.append(np.concatenate([center - half_width, center + half_width]))
    return np.array(boxes)


def relabel_pass1(traj, sampler, num_avoid, max_half_width):
    boxes = sample_boxes(sampler, num_avoid, max_half_width)
    traj['avoid_states'] = boxes
    traj['success'] = avoids_all(traj['observations'][:, :2], boxes)
    return traj['success']


def relabel_pass2(traj, paired_success, sampler, num_avoid, max_half_width):
    success = paired_success
    while success == paired_success:
        boxes = sample_boxes(sampler, num_avoid, max_half_width)
        success = avoids_all(traj['observations'][:, :2], boxes)
    traj['avoid_states'] = boxes
    traj['success'] = success
    return success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze', type=str, default='U_MAZE', choices=list(MAZE_NAMES))
    parser.add_argument('--num_avoid', type=int, default=3, help='number of relabeled avoid regions per trajectory')
    parser.add_argument('--max_episode_steps', type=int, default=None, help='episode length of the random-policy data (default: 300 for U_MAZE, 800 for A_MAZE and H_MAZE)')
    parser.add_argument('--max_half_width', type=float, default=0.2, help='maximum half-width of relabeled avoid boxes')
    parser.add_argument('--output', type=str, default=None, help='file name in RADT_DATASET_DIR')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    max_episode_steps = args.max_episode_steps or (300 if args.maze == 'U_MAZE' else 800)
    maze_tag = f'{MAZE_NAMES[args.maze]}{max_episode_steps}'
    output = args.output or f'pointmaze_obstacle_{maze_tag}_numavoid{args.num_avoid}_relabeled.pkl'

    # the random-policy data with one avoid region; its avoid-region features are dropped below
    trajectories = load_random_policy_data('PointMazeObstacle', f'buffer_numavoid1_{maze_tag}')
    copies = copy.deepcopy(trajectories)
    sampler = AccessibleRegionSampler(getattr(maps_obstacle, args.maze))

    successes1, successes2 = [], []
    for traj, traj2 in zip(trajectories, copies):
        success1 = relabel_pass1(traj, sampler, args.num_avoid, args.max_half_width)
        successes1.append(success1)
        successes2.append(relabel_pass2(traj2, success1, sampler, args.num_avoid, args.max_half_width))
        traj['observations'] = traj['observations'][:, :STATE_DIM]
        traj2['observations'] = traj2['observations'][:, :STATE_DIM]

    print('fraction of successful avoid demonstrations: pass 1 {:.3f}, pass 2 {:.3f}'.format(np.mean(successes1), np.mean(successes2)))
    save_dataset(trajectories + copies, output)
