# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from .trajectory_gpt2 import GPT2Model

'''
reach-avoid transformer:
input: [success indicator z, avoid start token, avoid tokens, goal start token, goal token, prompt end token] + (s,a,s,a,...)
predict: at each s, predict next a (and whether s violates an avoid region)
'''
class ReachAvoidTransformer(nn.Module):

    def __init__(
            self,
            state_dim,
            act_dim,
            action_space,
            prompt_dim, # dimension of the goal token
            hidden_size,
            max_length=None, # context len
            max_ep_len=None, # max timesteps for positional embedding
            adelta=0,
            avoid_rep='box', # 'box': [lower bounds, upper bounds] (2*prompt_dim), 'state': a single state (prompt_dim)
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.action_space = action_space
        self.discrete_action = True if hasattr(action_space, 'n') else False
        self.prompt_dim = prompt_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.n_positions = kwargs["n_positions"]
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, n_ctx=kwargs["n_positions"], **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_prompt = torch.nn.Linear(self.prompt_dim, hidden_size)
        # unused, kept so that checkpoints keep the same parameter names
        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if not self.discrete_action else []))
        )

        self.avoid_rep = avoid_rep
        avoid_token_dim = self.prompt_dim if avoid_rep == 'state' else 2 * self.prompt_dim
        self.embed_avoid_prompt = torch.nn.Linear(avoid_token_dim, hidden_size)
        self.embed_direction = torch.nn.Linear(2, hidden_size)
        self.embed_successfulness = torch.nn.Linear(2, hidden_size)
        self.adelta = adelta

        self.predict_success_state = torch.nn.Linear(hidden_size, 2)

        self.prompt_end_token = nn.Parameter(torch.randn(hidden_size) * config.initializer_range)

    def forward(self, states, actions, timesteps, attention_mask=None, prompt=None, avoid_prompt=None, success_list=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # convert action labels to onehot
        if self.discrete_action:
            actions = F.one_hot(actions, num_classes=self.act_dim).to(dtype=torch.float32)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # this makes the sequence look like (s_1, a_1, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # goal start token + goal token(s) + prompt end token; prompt tokens get the attention boost adelta
        prompt_goals, prompt_attention_mask = prompt
        batch = prompt_goals.shape[0]
        prompt_goal_embeddings = self.embed_prompt(prompt_goals) # (batch, prompt_len, dim)
        direction_positive = self.embed_direction(torch.tensor(np.array([np.array([[1, 0]])] * batch), device=prompt_goals.device, dtype=torch.float32))
        prompt_end = self.prompt_end_token.expand(batch, 1, -1)
        prompt_goal_embeddings = torch.cat((direction_positive, prompt_goal_embeddings, prompt_end), dim=1)
        boost = torch.tensor([[1 + self.adelta]] * batch, device = prompt_attention_mask.device, dtype=torch.float32)
        prompt_attention_mask = torch.cat((boost, (1 + self.adelta) * prompt_attention_mask, boost), dim=1)

        stacked_inputs = torch.cat((prompt_goal_embeddings, stacked_inputs), dim=1)
        stacked_attention_mask = torch.cat((prompt_attention_mask, stacked_attention_mask), dim=1)

        # avoid start token + avoid tokens (omitted entirely when avoid_prompt is None)
        if avoid_prompt is not None:
            avoid_prompt_states, avoid_prompt_attention_mask = avoid_prompt
            batch = avoid_prompt_states.shape[0]
            avoid_prompt_embeddings = self.embed_avoid_prompt(avoid_prompt_states) # (batch, avoid_prompt_len, dim)
            direction_negative = self.embed_direction(torch.tensor(np.array([np.array([[0, 1]])] * batch), device=avoid_prompt_states.device, dtype=torch.float32))
            avoid_prompt_embeddings = torch.cat((direction_negative, avoid_prompt_embeddings), dim=1)
            avoid_prompt_attention_mask = torch.cat((torch.tensor([[1 + self.adelta]] * batch, device = avoid_prompt_attention_mask.device, dtype=torch.float32), (1 + self.adelta) * avoid_prompt_attention_mask), dim=1)

            stacked_inputs = torch.cat((avoid_prompt_embeddings, stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((avoid_prompt_attention_mask, stacked_attention_mask), dim=1)

        # avoid success indicator token z
        successfulness_embedding = self.embed_successfulness(torch.tensor(np.array([np.array([[1, 0]]) if s else np.array([[0, 1]]) for s in success_list]), device=stacked_inputs.device, dtype=torch.float32))
        successfulness_mask = torch.tensor([[1 + self.adelta]] * batch_size, device = stacked_attention_mask.device, dtype=torch.float32)
        stacked_inputs = torch.cat((successfulness_embedding, stacked_inputs), dim=1)
        stacked_attention_mask = torch.cat((successfulness_mask, stacked_attention_mask), dim=1)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state'] # (batch, prompt_len+seq_len*2, dim)

        # split off the prompt part; x[:,0,t] is the token for s_t, x[:,1,t] for a_t
        x_orig = torch.clone(x)
        x = x[:, -seq_length*2:, :].reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        state_preds = self.predict_state(torch.concat([x_orig[:, :-seq_length*2], x[:,1]], axis=1))[:, -seq_length:, :]    # predict next state given state and action
        action_preds = self.predict_action(torch.concat([x_orig[:, :-seq_length*2], x[:,0]], axis=1))[:, -seq_length:, :]  # predict next action given state

        avoid_success_state_preds = self.predict_success_state(torch.concat([x_orig[:, :-seq_length*2], x[:,0]], axis=1))[:, -seq_length:, :]

        return state_preds, action_preds, avoid_success_state_preds

    def get_action(self, states, actions, timesteps, prompt=None, avoid_prompt=None, deterministic=True, success_list=None):
        states = states.reshape(1, -1, self.state_dim)
        if self.discrete_action:
            actions = actions.reshape(1, -1)
        else:
            actions = actions.reshape(1, -1, self.act_dim)
        timesteps = timesteps.reshape(1, -1)
        attention_mask = torch.ones(states.shape[1]).to(dtype=torch.long, device=states.device).reshape(1, -1)

        _, action_preds, _ = self.forward(states, actions, timesteps, attention_mask, prompt, avoid_prompt, success_list)

        if self.discrete_action:
            if deterministic:
                ret = torch.argmax(action_preds[0,-1], dim=-1)
            else:
                action_dist = torch.distributions.categorical.Categorical(logits=action_preds[0,-1])
                ret = action_dist.sample()
        else:
            if deterministic:
                ret = action_preds[0,-1]
            else:
                mean = action_preds[0,-1]
                std = torch.exp(-torch.ones_like(mean))
                action_dist = torch.distributions.Normal(mean, std)
                ret = action_dist.sample()
        return ret
