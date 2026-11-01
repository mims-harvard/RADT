'''
prompt utils for training & evaluation
'''

import numpy as np
import torch

from envs.common import get_goal_prompt, get_avoid_prompt


''' sample batch from trajectories dataset '''

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

# given a trajectory, convert it into Transformer sequence
def get_sequence(trajectory, K, info, gamma=1):
    state_dim, act_dim, device, discrete_action = info['state_dim'], info['act_dim'], info['device'], info['discrete_action']
    max_len = K

    s = trajectory['observations'].reshape(1, -1, state_dim)
    if discrete_action:
        a = trajectory['actions'].reshape(1, -1)
    else:
        a = trajectory['actions'].reshape(1, -1, act_dim)
    r = trajectory['rewards'].reshape(1, -1, 1)
    d = trajectory['terminals'].reshape(1, -1)
    timesteps = np.arange(0, trajectory['timesteps']).reshape(1, -1)
    rtg = discount_cumsum(trajectory['rewards'], gamma=gamma)[:s.shape[1] + 1].reshape(1, -1, 1)
    if rtg.shape[1] <= s.shape[1]:
        rtg = np.concatenate([rtg, np.zeros((1, 1, 1))], axis=1)

    # padding to the right
    tlen = s.shape[1]
    s = np.concatenate([s, np.zeros((1, max_len - tlen, state_dim))], axis=1)
    if discrete_action:
        a = np.concatenate([a, np.zeros((1, max_len - tlen))], axis=1)
    else:
        a = np.concatenate([a, np.zeros((1, max_len - tlen, act_dim))], axis=1)
    r = np.concatenate([r, np.zeros((1, max_len - tlen, 1))], axis=1)
    d = np.concatenate([d, np.ones((1, max_len - tlen)) * 2], axis=1)
    rtg = np.concatenate([rtg, np.zeros((1, max_len - tlen, 1))], axis=1)
    timesteps = np.concatenate([timesteps, np.zeros((1, max_len - tlen))], axis=1)
    mask = np.concatenate([np.ones((1, tlen)), np.zeros((1, max_len - tlen))], axis=1)

    s = torch.from_numpy(s).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(a).to(dtype=torch.long if discrete_action else torch.float32, device=device)
    r = torch.from_numpy(r).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(d).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(mask).to(device=device)

    return s, a, r, d, rtg, timesteps, mask


# from the dataset, sample a batch of (prompt, avoid prompt, sequence, avoid success indicator) for training
def get_prompt_batch(trajectories, info, variant, env_module):
    num_trajectories = len(trajectories)
    max_avoid_prompt_len = variant['max_avoid_prompt_len']
    device = info['device']
    K = variant['max_ep_len']  # context length = episode length
    prompt_dim = env_module.PROMPT_DIM
    avoid_token_dim = prompt_dim if env_module.AVOID_REP == 'state' else 2 * prompt_dim

    def fn(batch_size=variant['batch_size']):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True
        )
        prompt_list, p_mask_list = [], []
        avoid_prompt_list, avoid_p_mask_list, success_list = [], [], []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []

        for i in batch_inds:
            trajectory = trajectories[i]
            p, p_mask = get_goal_prompt(trajectory, prompt_dim, device=device)
            prompt_list.append(p)
            p_mask_list.append(p_mask)

            avoid_p, avoid_p_mask, success = get_avoid_prompt(trajectory, avoid_token_dim, max_avoid_prompt_len, device=device)
            avoid_prompt_list.append(avoid_p)
            avoid_p_mask_list.append(avoid_p_mask)
            success_list.append(success)

            s, a, r, d, rtg, timesteps, mask = get_sequence(trajectory, K, info)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)
            rtg_list.append(rtg)
            timesteps_list.append(timesteps)
            mask_list.append(mask)

        p, p_mask = torch.cat(prompt_list, dim=0), torch.cat(p_mask_list, dim=0)
        s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list, dim=0)
        rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)
        avoid_p, avoid_p_mask = torch.cat(avoid_prompt_list, dim=0), torch.cat(avoid_p_mask_list, dim=0)
        prompt = p, p_mask
        avoid_prompt = avoid_p, avoid_p_mask
        batch = s, a, r, d, rtg, timesteps, mask
        return prompt, avoid_prompt, batch, torch.tensor(success_list, device=device)

    return fn


""" evaluation """

def eval_episodes(model, env, env_module, info, variant):
    # MNC: mean over episodes of (number of steps inside an avoid region) / (episode length)
    # success rate: fraction of episodes that reach the goal
    normalized_costs, ep_lens, successes = [], [], []
    for _ in range(variant['num_eval_episodes']):
        with torch.no_grad():
            cost_ret, ep_len, success = prompt_evaluate_episode(env, env_module, model, info, variant)
        normalized_costs.append(cost_ret / ep_len)
        ep_lens.append(ep_len)
        successes.append(success)
    return {
        'mnc': np.mean(normalized_costs),
        'success_rate': np.mean(successes),
        'ep_len_mean': np.mean(ep_lens),
    }


def prompt_evaluate_episode(env, env_module, model, info, variant):
    state_dim, act_dim, device, discrete_action = info['state_dim'], info['act_dim'], info['device'], info['discrete_action']
    prompt_dim = env_module.PROMPT_DIM

    model.eval()
    model.to(device=device)

    obs, _ = env.reset()

    # goal prompt: the environment goal
    goal = obs["desired_goal"]
    prompt_goal = torch.from_numpy(np.array([goal]).reshape(1, -1, prompt_dim)).to(dtype=torch.float32, device=device)
    prompt_mask = torch.ones(prompt_goal.shape[:2]).to(device=device)
    prompt = (prompt_goal, prompt_mask)

    # avoid prompt: the environment's avoid regions
    avoid_tokens = env_module.eval_avoid_tokens(getattr(env, 'unwrapped', env), obs, variant)
    avoid_prompt_state = torch.from_numpy(avoid_tokens.reshape(1, len(avoid_tokens), -1)).to(dtype=torch.float32, device=device)
    avoid_prompt_mask = torch.ones(avoid_prompt_state.shape[:2]).to(device=device)
    avoid_prompt = (avoid_prompt_state, avoid_prompt_mask)

    state = obs["observation"][:state_dim] # remove avoid-region information

    # we keep all the histories on the device
    # note that the latest action will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    if discrete_action:
        actions = torch.zeros((0,), device=device, dtype=torch.long)
    else:
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_cost_return, episode_length, success = 0, 0, False
    for t in range(variant['max_ep_len']):
        if discrete_action:
            actions = torch.cat([actions, torch.zeros((1,), device=device, dtype=torch.long)], dim=0)
        else:
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        # always condition on avoid success z = 1
        action = model.get_action(
            states.to(dtype=torch.float32),
            actions,
            timesteps.to(dtype=torch.long),
            prompt=prompt,
            avoid_prompt=avoid_prompt,
            success_list=torch.tensor([True], device=device)
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()

        obs, reward, cost, done, infos = env.step(action)
        state = obs["observation"][:state_dim]

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_cost_return += cost
        episode_length += 1
        success = infos["is_success"]

        if done or infos["is_success"]: # stop when success is reached
            break

    return episode_cost_return, episode_length, success
