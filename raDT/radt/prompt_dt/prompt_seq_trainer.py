# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import os
import numpy as np
import torch
import torch.nn.functional as F
import time


def check_avoid_success_state(states, avoid_tokens, avoid_mask, obs_start_index = 0, prompt_dim=3, avoid_rep='box'):
    """
    states: B x seq_length x dim
    avoid_tokens: B x num_avoid x 2*dim for 'box' (lower bounds, upper bounds), B x num_avoid x dim for 'state'
    avoid_mask: B x num_avoid (ignored for now because the padding is done with the initial avoid box in the prompt, not affecting the boolean final result)
    returns: B x seq_length, True where the state is outside all avoid regions
    """
    states = states[:,:, obs_start_index : obs_start_index + prompt_dim]
    if avoid_rep == 'state':
        a = (avoid_tokens.unsqueeze(1) == states.unsqueeze(2)).all(dim=-1)
        return ~a.any(dim=2)
    avoid_boxes_upper, avoid_boxes_lower = avoid_tokens[:, :, prompt_dim:], avoid_tokens[:, :, :prompt_dim]
    a = (avoid_boxes_lower.unsqueeze(1) <= states.unsqueeze(2)) & (avoid_boxes_upper.unsqueeze(1) >= states.unsqueeze(2))
    result = ~a.any(dim=2).all(dim=2)
    return result


class PromptSequenceReachAvoidTrainer:

    def __init__(self, model, optimizer, batch_size,
                 scheduler=None,
                 get_prompt_batch=None,
                 alpha1=1,
                 alpha2=1,
                 obs_start_index=0,
                 prompt_dim=3,
                 avoid_rep='box'):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        if model.discrete_action:
            self.loss_fn = F.cross_entropy
        else:
            self.loss_fn = lambda a_hat, a: torch.mean((a_hat - a) ** 2)
        self.scheduler = scheduler
        self.diagnostics = dict()
        self.get_prompt_batch = get_prompt_batch
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.obs_start_index = obs_start_index
        self.prompt_dim = prompt_dim
        self.avoid_rep = avoid_rep

        self.start_time = time.time()


    def train(self, num_steps):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/lr'] = self.scheduler.get_lr()[0]

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs

    def train_step(self):
        prompt, avoid_prompt, batch, success_list = self.get_prompt_batch()
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch
        action_target = torch.clone(actions)
        state_preds, action_preds, success_preds = self.model.forward(states, actions, timesteps, attention_mask=attention_mask, prompt=prompt, avoid_prompt=avoid_prompt, success_list=success_list)

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        if self.model.discrete_action:
            action_target = action_target.reshape(-1)[attention_mask.reshape(-1) > 0]
        else:
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        action_loss = self.loss_fn(action_preds, action_target)

        # avoid awareness loss: predict at each timestep whether the state is outside all avoid regions
        avoid_prompt_states, avoid_prompt_attention_mask = avoid_prompt
        success_target = check_avoid_success_state(states, avoid_prompt_states, avoid_prompt_attention_mask, self.obs_start_index, prompt_dim = self.prompt_dim, avoid_rep = self.avoid_rep) # B x seq_length true/false

        success_target = torch.stack([success_target.float(), (~success_target).float()], axis=2) # B x seq_length x 2 one-hot
        success_target = success_target.reshape(-1, 2)[attention_mask.reshape(-1) > 0]
        success_preds = success_preds.reshape(-1, 2)[attention_mask.reshape(-1) > 0]

        success_accuracy = (success_preds.argmax(axis=1) == success_target.argmax(axis=1)).float().mean()
        neg_indices = (success_target[:, 1] == 1).bool()
        success_target_neg = success_target[neg_indices]
        success_preds_neg = success_preds[neg_indices]
        success_accuracy_neg = (success_preds_neg.argmax(axis=1) == success_target_neg.argmax(axis=1)).float().mean()
        pos_indices = (success_target[:, 0] == 1).bool()
        success_target_pos = success_target[pos_indices]
        success_preds_pos = success_preds[pos_indices]
        success_accuracy_pos = (success_preds_pos.argmax(axis=1) == success_target_pos.argmax(axis=1)).float().mean()
        prop_pos = success_target_pos.shape[0] / success_target.shape[0]

        success_loss_neg = F.cross_entropy(success_preds_neg, success_target_neg)
        success_loss_pos = F.cross_entropy(success_preds_pos, success_target_pos)

        success_loss = success_loss_pos + self.alpha2 * success_loss_neg
        loss = action_loss + self.alpha1 * success_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        self.diagnostics['training/total_error'] = loss.detach().cpu().item()
        self.diagnostics['training/action_error'] = action_loss.detach().cpu().item()
        self.diagnostics['training/success_error'] = success_loss.detach().cpu().item()
        self.diagnostics['training/success_accuracy'] = success_accuracy.detach().cpu().item()
        self.diagnostics['training/success_accuracy_neg'] = success_accuracy_neg.detach().cpu().item()
        self.diagnostics['training/success_accuracy_pos'] = success_accuracy_pos.detach().cpu().item()
        self.diagnostics['training/prop_pos'] = prop_pos

        return loss.detach().cpu().item()


    def save_model(self, env_name, postfix, folder):
        path = os.path.join(folder, 'prompt_model_' + env_name + postfix)
        torch.save(self.model.state_dict(), path)
        print('model saved to ', path)
