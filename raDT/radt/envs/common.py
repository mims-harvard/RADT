import pickle
import numpy as np
import torch


def load_trajectories(dataset_path):
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)


def get_goal_prompt(trajectory, prompt_dim, device=None):
    # hindsight goal: the state reached at the end of the trajectory
    goal = trajectory['next_observations'][trajectory['timesteps'] - 1][:prompt_dim]
    prompt = torch.from_numpy(np.array(goal).reshape(1, 1, prompt_dim)).to(dtype=torch.float32, device=device)
    mask = torch.ones((1, 1), device=device, dtype=torch.float64)
    return prompt, mask


def get_avoid_prompt(trajectory, token_dim, max_avoid_prompt_length=10, device=None):
    # hindsight-relabeled avoid regions, left-padded (with the first avoid region) to max_avoid_prompt_length
    avoid_states = np.asarray(trajectory["avoid_states"])
    if avoid_states.ndim == 2:
        prompt_length = min(avoid_states.shape[0], max_avoid_prompt_length)
    else:
        prompt_length = 0
    timesteps = [0] * (max_avoid_prompt_length - prompt_length) + list(np.arange(prompt_length))
    mask = np.concatenate([np.zeros((1, max_avoid_prompt_length - prompt_length)), np.ones((1, prompt_length))], axis=1)

    if prompt_length > 0:
        prompt = [avoid_states[t, :] for t in timesteps]
    else:
        prompt = [np.zeros(token_dim) for _ in timesteps]
    prompt = np.array(prompt).reshape(1, -1, token_dim)

    prompt = torch.from_numpy(prompt).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(device=device)
    return prompt, mask, trajectory["success"]


def box_tokens(centers, half_width):
    # [lower bounds, upper bounds] for each center
    return np.array([np.concatenate([c - half_width, c + half_width]) for c in centers])
