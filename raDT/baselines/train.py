import os
import argparse
import random
import numpy as np
import gymnasium as gym
import torch
import wandb
from mpi4py import MPI

from raDT.baselines.envs import register_envs, maps_obstacle
from raDT.baselines.rl_modules.cactionablemodel_agent import ActionableModel
from raDT.baselines.rl_modules.gcsl_agent import GCSL
from raDT.baselines.rl_modules.rbsl_agent import RBSL
from raDT.constants import CHECKPOINT_DIR, WANDB_DIR, WANDB_ENTITY, WANDB_PROJECT_BASELINES

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
"""

MAZE_NAMES = {'U_MAZE': 'umaze', 'A_MAZE': 'amaze', 'H_MAZE': 'hmaze'}


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env', type=str, default='FetchReachObstacle', choices=['FetchReachObstacle', 'PointMazeObstacle'])
    parser.add_argument('--method', default='AMlag', type=str, choices=['AMlag', 'rbsl', 'wgcsl'])
    parser.add_argument('--n-epochs', type=int, default=500, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=20, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=20, help='the times to update the network')
    parser.add_argument('--n-safety-batches', type=int, default=30, help='the times to update the safety network')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--safety-replay-strategy', type=str, default='recovery', help='the recovery strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default=os.path.join(CHECKPOINT_DIR, 'baselines'), help='the path to save the models')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--noise-eps', type=float, default=1.0, help='noise eps')
    parser.add_argument('--buffer-size', type=int, default=int(2e6), help='the size of the buffer')
    parser.add_argument('--safety-buffer-size', type=int, default=int(2e6), help='the size of the safety buffer')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=512, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--lr-safety-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-safety-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=60, help='the number of evaluation episodes after each epoch')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--cuda', default=True, type=boolean, help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=1, help='the rollouts per mpi')

    parser.add_argument('--eval', default=True, type=boolean, help='evaluate (and save a checkpoint) after every epoch')
    parser.add_argument('--seed', type=int, default=0, help='random seed of the first run')
    parser.add_argument('--n_seeds', type=int, default=1, help='number of runs, with seeds seed, seed+1, ...')

    parser.add_argument('--relabel', default=True, type=boolean)
    parser.add_argument('--safety_relabel_percent', default=0.5, type=float)
    parser.add_argument('--reward_type', default='binary', type=str)
    parser.add_argument('--threshold', default=0.05, type=float)

    # offline data
    parser.add_argument('--expert_percent', type=float, default=0, help='fraction of the training data from the expert policy')
    parser.add_argument('--random_percent', type=float, default=1, help='fraction of the training data from the random policy')
    parser.add_argument('--massless', type=int, default=1, help='FetchReachObstacle: use the dataset where the avoid box can be passed through')
    parser.add_argument('--buffer_suffix', type=str, default='', help='FetchReachObstacle: select a dataset with recomputed costs, e.g. _box0.20 (see recompute_cost.py)')

    # environment
    parser.add_argument('--bsa_box_size', type=float, default=None, help='FetchReachObstacle: half of the avoid-box width used for evaluation')
    parser.add_argument('--maze', type=str, default='U_MAZE', choices=list(MAZE_NAMES), help='PointMazeObstacle: the maze')
    parser.add_argument('--num_avoid', type=int, default=1, help='PointMazeObstacle: number of avoid regions')
    parser.add_argument('--max_episode_steps', type=int, default=None, help='PointMazeObstacle: steps per episode (300 for U_MAZE, 800 for A_MAZE and H_MAZE)')

    parser.add_argument('--group', type=str, default=None, help='wandb group name')
    parser.add_argument('--run_suffix', type=str, default="", help='run suffix')
    parser.add_argument('--no_wandb', action='store_true', default=False)

    args = parser.parse_args()
    args.maze_name = MAZE_NAMES[args.maze]
    return args

def get_env_params(env):
    obs, info = env.reset(seed = args.seed + MPI.COMM_WORLD.Get_rank())
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'action_space': env.action_space
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def get_method_params(args):
    if args.method == 'AMlag':
        args.lr_actor = 0.001
        args.lr_critic = 0.001
    else:
        args.lr_actor = 5e-4
        args.lr_critic = 5e-4

    # fraction of hindsight-relabeled goals
    if args.method == 'AMlag':
        args.relabel_percent = 0.5
    elif args.method == 'rbsl' and args.env == 'PointMazeObstacle' and args.expert_percent in (0.5, 1):
        args.relabel_percent = 0
    else:
        args.relabel_percent = 1.0
    if not args.relabel:
        args.relabel_percent = 0.
    args.use_disc = False


def launch(args):

    get_method_params(args)
    args.env_id = f'{args.env}-v0'
    if args.env == 'PointMazeObstacle':
        if args.max_episode_steps is None:
            args.max_episode_steps = 300 if args.maze == 'U_MAZE' else 800
        env = gym.make(args.env_id, maze_map=getattr(maps_obstacle, args.maze), num_avoid=args.num_avoid, max_episode_steps=args.max_episode_steps)
    else:
        if args.bsa_box_size:
            env = gym.make(args.env_id, bsa_box_size=args.bsa_box_size)
        else:
            env = gym.make(args.env_id)

    relabel_tag = f'relabel{args.relabel_percent}'
    reward_tag = f'{args.reward_type}{args.threshold}' if args.reward_type == 'binary' else args.reward_type
    if args.env == 'PointMazeObstacle':
        run_name = f'{args.env_id}-{args.expert_percent}-{args.random_percent}-{args.method}-{reward_tag}-{relabel_tag}-{args.seed}-{args.maze_name}{args.max_episode_steps}-num{args.num_avoid}{args.run_suffix}'
    else:
        run_name = f'{args.env_id}-{args.expert_percent}-{args.random_percent}-{args.method}-{reward_tag}-{relabel_tag}-{args.seed}-m{args.massless}-bsa{args.bsa_box_size}{args.buffer_suffix}{args.run_suffix}'
    args.run_name = run_name

    if MPI.COMM_WORLD.Get_rank() == 0:
        os.makedirs(WANDB_DIR, exist_ok=True)
        wandb.init(project=WANDB_PROJECT_BASELINES, entity=WANDB_ENTITY, dir=WANDB_DIR, name=run_name,
                   group=args.group, config=args, mode='disabled' if args.no_wandb else None)

    # set random seeds for reproduce
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get the environment parameters
    env_params = get_env_params(env)

    # create agent
    if args.method == 'wgcsl':
        trainer = GCSL(args, env, env_params)
    elif args.method == 'AMlag':
        trainer = ActionableModel(args, env, env_params)
    elif args.method == 'rbsl':
        trainer = RBSL(args, env, env_params)
    print(run_name)

    # do offline goal-conditioned rl
    result = trainer.learn(evaluate_agent=args.eval)

    if MPI.COMM_WORLD.Get_rank() == 0:
        wandb.finish()
    return result

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    args = get_args()
    register_envs()

    first_seed = args.seed
    for seed in range(first_seed, first_seed + args.n_seeds):
        args.seed = seed
        launch(args)
