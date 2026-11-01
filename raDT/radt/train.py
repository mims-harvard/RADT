import os
import ast
import argparse
import importlib
import random
from datetime import datetime

import numpy as np
import torch
import wandb
from tqdm import trange

from prompt_dt.prompt_decision_transformer import ReachAvoidTransformer
from prompt_dt.prompt_seq_trainer import PromptSequenceReachAvoidTrainer
from prompt_dt.prompt_utils import eval_episodes, get_prompt_batch
from envs.common import load_trajectories
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from raDT.constants import RADT_DATASET_DIR, CHECKPOINT_DIR, WANDB_DIR, WANDB_ENTITY, WANDB_PROJECT_RADT

ENVS = ['reach_obstacle', 'pointmaze_obstacle', 'cardiogenesis']


def experiment(variant):
    device = variant['device']
    env_module = importlib.import_module(f"envs.{variant['env']}")

    if variant['seed'] is not None:
        random.seed(variant['seed'])
        np.random.seed(variant['seed'])
        torch.manual_seed(variant['seed'])

    # evaluation environment
    env = env_module.make_env(variant)
    state_dim = env_module.STATE_DIM
    act_dim = env.action_space.n if env_module.DISCRETE_ACTION else env.action_space.shape[0]
    info = {'state_dim': state_dim, 'act_dim': act_dim, 'device': device, 'discrete_action': env_module.DISCRETE_ACTION}

    ######
    # model
    ######
    max_ep_len = variant['max_ep_len']
    model = ReachAvoidTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        action_space=env.action_space,
        prompt_dim=env_module.PROMPT_DIM,
        max_length=max_ep_len,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=variant['n_positions'],
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        adelta=variant['adelta'],
        avoid_rep=env_module.AVOID_REP,
    )
    model = model.to(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )

    # schedulers step once per gradient step (num_steps_per_iter gradient steps per iteration)
    warmup_steps = variant['warmup_steps']
    if variant['scheduler'] == 'lambdalr':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / warmup_steps, 1)
        )
    elif variant['scheduler'] == 'cosinewarmuprestarts':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=variant['T_0'],
            cycle_mult=variant['T_mult'],
            max_lr=variant['learning_rate'],
            min_lr=variant['learning_rate_min'],
            warmup_steps=warmup_steps,
            gamma=variant['cosine_gamma'],
        )

    if variant['evaluation']:
        ######
        # evaluate a trained model
        ######
        load_path = os.path.join(CHECKPOINT_DIR, variant['load_path'])
        model.load_state_dict(torch.load(load_path, map_location=torch.device(device)), strict=True)
        print('model initialized from: ', load_path)
        logs = eval_episodes(model, env, env_module, info, variant)
        print('=' * 80)
        for k, v in logs.items():
            print(f'evaluation/{k}: {v}')
        return

    ######
    # train
    ######
    trajectories = load_trajectories(os.path.join(RADT_DATASET_DIR, variant['dataset_path']))
    assert trajectories[0]['observations'].shape[1] == state_dim

    dataset_name = os.path.splitext(os.path.basename(variant['dataset_path']))[0]
    time_now = datetime.now().strftime("%Y%m%d%H%M%S")
    group_name = f"{variant['exp_name']}-{variant['env']}-{dataset_name}"
    run_name = f"{group_name}-{variant['instance_prefix']}-{time_now}"

    trainer = PromptSequenceReachAvoidTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=variant['batch_size'],
        scheduler=scheduler,
        get_prompt_batch=get_prompt_batch(trajectories, info, variant, env_module),
        alpha1=variant['alpha1'],
        alpha2=variant['alpha2'],
        obs_start_index=env_module.OBS_START_INDEX,
        prompt_dim=env_module.PROMPT_DIM,
        avoid_rep=env_module.AVOID_REP,
    )

    save_dir = os.path.join(CHECKPOINT_DIR, 'radt', run_name)
    os.makedirs(save_dir, exist_ok=True)
    if variant['log_to_wandb']:
        os.makedirs(WANDB_DIR, exist_ok=True)
        wandb.init(
            name=run_name,
            group=group_name,
            project=WANDB_PROJECT_RADT,
            entity=WANDB_ENTITY,
            dir=WANDB_DIR,
            config=variant
        )

    for iter in trange(variant['max_iters']):
        outputs = trainer.train(num_steps=variant['num_steps_per_iter'])

        if iter % variant['eval_interval'] == 0 and iter > 0:
            eval_logs = eval_episodes(model, env, env_module, info, variant)
            print('=' * 80)
            print(f'Iteration {iter}')
            for k, v in eval_logs.items():
                outputs[f'evaluation/{k}'] = v
                print(f'evaluation/{k}: {v}')

        if iter % variant['save_interval'] == 0:
            trainer.save_model(env_name=variant['env'], postfix='_iter_' + str(iter), folder=save_dir)

        outputs.update({"global_step": iter})
        if variant['log_to_wandb']:
            wandb.log(outputs)

    trainer.save_model(env_name=variant['env'], postfix='_iter_' + str(iter), folder=save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='radt')
    parser.add_argument('--instance_prefix', type=str, default='run', help='name of this run within the experiment group')
    parser.add_argument('--env', type=str, default='reach_obstacle', choices=ENVS)
    parser.add_argument('--dataset_path', type=str, default=None, help='training dataset file, relative to RADT_DATASET_DIR in raDT/constants.py')
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--evaluation', action='store_true', default=False, help='evaluate the model at --load-path instead of training')
    parser.add_argument('--load-path', type=str, default=None, help='model checkpoint, relative to CHECKPOINT_DIR in raDT/constants.py')

    parser.add_argument('--max_avoid_prompt_len', type=int, default=10) # max number of avoid tokens during training
    parser.add_argument('--max_ep_len', type=int, default=50) # max episode length in both dataset & env; also the context length
    parser.add_argument('--adelta', type=float, default=0.0, help='attention boosting for the prompt tokens')
    parser.add_argument('--alpha1', type=float, default=1, help='weight of the avoid awareness loss relative to the action loss')
    parser.add_argument('--alpha2', type=float, default=1.1, help='weight of the in-avoid-region class in the avoid awareness loss')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--n_positions', type=int, default=2048, help='max transformer context length (prompt + 2 * max_ep_len tokens)')
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)

    # all step counts below are in gradient steps, except max_iters / eval_interval / save_interval,
    # which count iterations of num_steps_per_iter gradient steps
    parser.add_argument('--scheduler', type=str, default='lambdalr', choices=['lambdalr', 'cosinewarmuprestarts'])
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--T_0', type=int, default=10000, help='cosinewarmuprestarts cycle length')
    parser.add_argument('--T_mult', type=int, default=1)
    parser.add_argument('--learning_rate_min', type=float, default=0)
    parser.add_argument('--cosine_gamma', type=float, default=1)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--num_eval_episodes', type=int, default=60)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_wandb', dest='log_to_wandb', action='store_false', default=True)

    # FetchReachObstacle
    parser.add_argument('--bsa_box_size', type=float, default=0.08, help='half of the avoid-box width')
    # MazeObstacle
    parser.add_argument('--maze', type=str, default='U_MAZE', choices=['U_MAZE', 'A_MAZE', 'H_MAZE'])
    parser.add_argument('--num_avoid', type=int, default=1, help='number of avoid regions in the maze')
    # Cardiogenesis
    parser.add_argument('--fixed_interval', type=int, default=10, help='asynchronous Boolean network updates per step (k)')
    parser.add_argument('--avoids_readable', type=str, default="[]", help="avoid states, e.g. \"['100000000001100']\"")
    parser.add_argument('--fixed_goal_readable', type=str, default=None)
    parser.add_argument('--fixed_start_readable', type=str, default=None)

    args = parser.parse_args()
    if not args.evaluation and args.dataset_path is None:
        parser.error('--dataset_path is required for training')
    args.avoids_readable = ast.literal_eval(args.avoids_readable)
    experiment(variant=vars(args))
