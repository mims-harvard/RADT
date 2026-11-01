from raDT.baselines.envs.reach_obstacle import FetchReachObstacleEnv
from .common import box_tokens

PROMPT_DIM = 3          # gripper position (goal token, avoid-box coordinates)
STATE_DIM = 10          # FetchReach observation without the obstacle features
OBS_START_INDEX = 0
AVOID_REP = 'box'
DISCRETE_ACTION = False


def make_env(args):
    # avoid box of width 2 * bsa_box_size centered on the obstacle
    return FetchReachObstacleEnv(bsa_box_size=args['bsa_box_size'])


def eval_avoid_tokens(env, obs, args):
    # the obstacle position is appended to the observation after the first STATE_DIM entries
    center = obs["observation"][STATE_DIM:STATE_DIM + PROMPT_DIM]
    return box_tokens([center], args['bsa_box_size'])
