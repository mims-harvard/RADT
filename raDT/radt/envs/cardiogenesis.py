import numpy as np
from gymnasium.spaces import Discrete
from pyboolnet.file_exchange import bnet2primes
from pyboolnet.state_transition_graphs import random_walk
from pyboolnet.attractors import compute_attractors

PROMPT_DIM = 15         # gene expression state (goal token, avoid token)
STATE_DIM = 15
OBS_START_INDEX = 0
AVOID_REP = 'state'     # avoid tokens are the avoid states themselves
DISCRETE_ACTION = True

# Boolean network model of mouse cardiogenesis (Herrmann et al., 2012)
GRN = """
    Bmp2,   (!canWnt & exogen_Bmp2_II)
    canWnt,    exogen_canWnt_II
    Dkk1,   (Mesp1 | (canWnt & !exogen_Bmp2_II))
    Fgf8,   (!Mesp1 & (Foxc1_2 | Tbx1))
    Foxc1_2,    (canWnt & exogen_canWnt_II)
    GATAs,    (Nkx2_5 | Mesp1 | Tbx5)
    Isl1,    (Tbx1 | Mesp1 | Fgf8 | (canWnt & exogen_canWnt_II))
    Mesp1,    (canWnt & !exogen_Bmp2_II)
    Nkx2_5,    ((Isl1 & GATAs) | Tbx1 | (Mesp1 & Dkk1) | (Bmp2 & GATAs) | Tbx5)
    Tbx1,    Foxc1_2
    Tbx5,    (!(Tbx1 | canWnt) & (Nkx2_5 | Tbx5 | Mesp1) & !(Dkk1 & !(Mesp1 | Tbx5)))
    exogen_Bmp2_I,    exogen_Bmp2_I
    exogen_Bmp2_II,    exogen_Bmp2_I
    exogen_canWnt_I,    exogen_canWnt_I
    exogen_canWnt_II,    exogen_canWnt_I
    """

# the six asynchronous attractors of the network
ATTRACTORS = ['000000000000000', '000010010100000', '001111011011111',
              '010111111010011', '100000000001100', '100010010101100']
FHF_STATE = '100010010101100'


def to_array(state_readable):
    return np.array([int(x) for x in state_readable])


def to_readable(state):
    return "".join(str(int(x)) for x in state)


class CardiogenesisEnv:
    """
    One step = flip the expression of one gene (the action), then run `fixed_interval`
    asynchronous Boolean network updates. Start and goal states are attractors unless fixed.
    """
    def __init__(self, fixed_interval=1, avoids_readable=[], fixed_goal_readable=None, fixed_start_readable=None, initial_state_list_readable=None):
        self.bnet = bnet2primes(GRN)
        if not initial_state_list_readable:
            initial_state_list_readable = [a["state"]['str'] for a in compute_attractors(self.bnet, "asynchronous")["attractors"]]
        self.initial_state_readable_list = initial_state_list_readable
        self.initial_state_matrix = np.array([to_array(s) for s in initial_state_list_readable])
        self.fixed_interval = fixed_interval
        self.fixed_avoids_readable = list(avoids_readable)
        self.avoid_states_readable_list = list(avoids_readable)
        self.fixed_goal_readable = fixed_goal_readable
        self.fixed_goal = to_array(fixed_goal_readable) if fixed_goal_readable else None
        self.fixed_start_readable = fixed_start_readable
        self.fixed_start = to_array(fixed_start_readable) if fixed_start_readable else None
        self.action_space = Discrete(15)
        self.reset()

    def _sample_goal(self):
        if self.fixed_goal_readable:
            self.goal = self.fixed_goal
            self.goal_readable = self.fixed_goal_readable
        else:
            random_i = np.random.choice(np.arange(len(self.initial_state_readable_list)))
            self.goal = self.initial_state_matrix[random_i, :]
            self.goal_readable = to_readable(self.goal)
        return self.goal, self.goal_readable

    def _sample_start(self):
        if self.fixed_start_readable:
            self.start = self.fixed_start
            self.start_readable = self.fixed_start_readable
        else:
            random_i = np.random.choice(np.arange(len(self.initial_state_readable_list)))
            self.start = self.initial_state_matrix[random_i, :]
            self.start_readable = to_readable(self.start)
        return self.start, self.start_readable

    def _get_obs(self):
        return {"observation": self.current_state.copy(),
                "achieved_goal": self.current_state.copy(),
                "desired_goal": self.goal.copy()}

    def step(self, action_int):
        self.ep += 1
        current_state_readable = self.current_state_readable
        perturbed_state_readable = current_state_readable[:action_int] + str(1 - int(current_state_readable[action_int])) + current_state_readable[action_int + 1:]
        path = random_walk(self.bnet, "asynchronous", initial_state=perturbed_state_readable, length=self.fixed_interval)
        new_state_readable = "".join([str(x) for x in path[-1].values()])

        self.current_state_readable = new_state_readable
        self.current_state = to_array(new_state_readable)

        reward = 0
        cost = int(self.current_state_readable in self.avoid_states_readable_list)
        done = False
        info = {"is_success": int(self.current_state_readable == self.goal_readable)}
        return self._get_obs(), reward, cost, done, info

    def reset(self):
        self.ep = 0
        self._sample_goal()
        self._sample_start()
        while self.goal_readable == self.start_readable:
            self._sample_goal()
            self._sample_start()

        self.current_state = self.start
        self.current_state_readable = self.start_readable
        info = {"is_success": int(self.current_state_readable == self.goal_readable)}
        return self._get_obs(), info


def make_env(args):
    return CardiogenesisEnv(fixed_interval=args['fixed_interval'],
                            avoids_readable=args['avoids_readable'],
                            fixed_goal_readable=args['fixed_goal_readable'],
                            fixed_start_readable=args['fixed_start_readable'])


def eval_avoid_tokens(env, obs, args):
    # prompt with the given avoid states; if none are given, avoid a random attractor other than the goal
    if env.fixed_avoids_readable:
        env.avoid_states_readable_list = list(env.fixed_avoids_readable)
    else:
        env.avoid_states_readable_list = [np.random.choice([s for s in ATTRACTORS if s != env.goal_readable])]
    return np.array([to_array(s) for s in env.avoid_states_readable_list])
