from raDT.baselines.envs.point_maze_obstacle import PointMazeObstacleEnv
from raDT.baselines.envs import maps_obstacle
from .common import box_tokens

PROMPT_DIM = 2          # agent xy position (goal token, avoid-box coordinates)
STATE_DIM = 4           # PointMaze observation without the avoid-region centroids
OBS_START_INDEX = 0
AVOID_REP = 'box'
DISCRETE_ACTION = False

# avoid regions are circles of radius 0.2; RADT is prompted with the circumscribing box
AVOID_HALF_WIDTH = 0.2


def make_env(args):
    maze_map = getattr(maps_obstacle, args['maze'])
    return PointMazeObstacleEnv(maze_map=maze_map, num_avoid=args['num_avoid'])


def eval_avoid_tokens(env, obs, args):
    # avoid-region centroids are appended to the observation after the first STATE_DIM entries
    state = obs["observation"]
    centers = [state[STATE_DIM + PROMPT_DIM * i: STATE_DIM + PROMPT_DIM * (i + 1)] for i in range(args['num_avoid'])]
    return box_tokens(centers, AVOID_HALF_WIDTH)
