import numpy as np
import scipy.signal
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torch.distributions as torch_dist

from simpl.nn import ToDeviceMixin


def discount(x, gamma, terminated=None, bootstrap_truncated=None):
    # y[n] = 1/a[0]  X  ( b[0] X x[n] - a[1]  X y[n-1] )
    # g[n]=               1   X r[n]  + gamma X g[n-1]
    if bootstrap_truncated is None or terminated:
        return scipy.signal.lfilter(b=[1], a=[1, -gamma], x=x[..., ::-1])[..., ::-1].copy()
    
    if bootstrap_truncated == 'last':
        bootstrap_reward = x[-1]
    elif bootstrap_truncated == 'mean':
        bootstrap_reward = x.mean(0)
    else:
        raise
    x = np.concatenate([x, [mock_reward / (1 - gamma)]], axis=-1)
    return scipy.signal.lfilter(b=[1], a=[1, -gamma], x=x[..., ::-1])[..., ::-1][:-1].copy()


class BatchEpisode(torch_data.Dataset):
    def __init__(self, episodes, max_context, context_l):
        self.episodes = episodes
        self.device = None

        # ha...
        n_context = int(min(max_context, np.ceil(max([len(episode)+1 for episode in episodes]) / context_l)))

        all_stacked_states = []
        all_firsts = []
        all_last_indices = []
        for episode in episodes:
            all_states = torch.stack(episode.processed_states, dim=0)
            for t in range(len(episode)+1):
                seq_state = all_states[max(0, t+1-n_context*context_l):t+1]
                seq_state = torch.cat([
                    seq_state,
                    torch.zeros(context_l - ((len(seq_state)-1) % context_l+1), *seq_state.shape[1:])
                ], dim=0)  # align to context boundary
                seq_state = torch.cat([
                    torch.zeros(n_context*context_l - len(seq_state), *seq_state.shape[1:]),
                    seq_state
                ], dim=0)  # pad front
                all_stacked_states.append(seq_state)

                first_context_i = n_context - 1 - t // context_l
                firsts = [
                    context_i <= first_context_i
                    for context_i in range(n_context)
                ]
                all_firsts.append(firsts)

                all_last_indices.append(t % context_l)

        all_stacked_states = torch.stack(all_stacked_states)
        self.all_stacked_states = all_stacked_states
        self.all_firsts = torch.as_tensor(np.array(all_firsts, dtype=bool))
        self.all_last_indices = torch.as_tensor(all_last_indices)

        action_keys = episodes[0].action_masks[0].keys()
        self.all_action_masks = {
            k: torch.cat([
                torch.stack([mask[k] for mask in episode.action_masks])
                for episode in episodes
            ])
            for k in action_keys
        }

        firsts = torch.cat([
            torch.tensor([True] + [False]*(len(episode)-1))
            for episode in episodes
        ])
        self.curr_indices = torch.arange(len(firsts)) + firsts.cumsum(-1) - 1
        self.curr_mask = torch.cat([
            torch.tensor([True]*len(episode) + [False])
            for episode in episodes
        ])
        self.next_mask = torch.cat([
            torch.tensor([False] + [True]*len(episode))
            for episode in episodes
        ])

        self.rewards = torch.as_tensor(
            np.concatenate([episode.rewards for episode in episodes])
        )

        # only for iterating along with all_states
        action_keys = episodes[0].actions[0].keys()
        self.all_actions = {
            k: torch.cat([
                torch.stack([a[k] for a in episode.actions + [episode.actions[-1]]])
                for episode in episodes
            ])
            for k in action_keys
        }
        self.actions = {
            k: v[self.curr_mask]
            for k, v in self.all_actions.items()
        }

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        ret = SimpleNamespace()
        all_state_idx = self.curr_indices[idx]
        ret.states = self.all_stacked_states[all_state_idx]
        ret.action_masks = {
            k: self.all_action_masks[k][all_state_idx]
            for k in self.all_action_masks
        }
        ret.firsts = self.all_firsts[all_state_idx]
        ret.last_indices = self.all_last_indices[all_state_idx]
        ret.actions = {
            k: self.actions[k][idx]
            for k in self.actions
        }
        ret.rewards = self.rewards[idx]
        if hasattr(self, 'values'):
            ret.values = self.values[idx]
        if hasattr(self, 'gaes'):
            ret.gaes = self.gaes[idx]
        if hasattr(self, 'init_log_probs'):
            ret.init_log_probs = self.init_log_probs[idx]
        return ret

    def compute_return(self, gamma, bootstrap_truncated=None):
        raise NotImplementedError
        returns = [
            discount(np.array(episode.rewards), gamma, episode.dones[-1], bootstrap_truncated)
            for episode in self.episodes
        ]
        self.data['returns'] = torch.tensor(
            np.concatenate(returns),
            dtype=torch.float32, device=self.device
        )
        self.data['all_returns'] = torch.tensor(
            np.concatenate([np.concatenate([ret, [0]]) for ret in returns]),
            dtype=torch.float32, device=self.device
        )

    def compute_gae(self, all_vs, gamma, gae_lambda, process_batch_size=None):
        vs = all_vs[self.curr_mask]
        next_vs = all_vs[self.next_mask]
        deltas = self.rewards + gamma*next_vs - vs
        gaes = discount(deltas.cpu().numpy(), gamma*gae_lambda)
        gaes = torch.as_tensor(gaes, dtype=torch.float32, device=self.device)

        self.values = vs
        self.gaes = gaes

    def to(self, device):
        raise
        self.device = device
        for k in self.data:
            self.data[k] = self.data[k].to(device)
        return self


class PPO(ToDeviceMixin, nn.Module):
    def __init__(self, policy, fixed_policy, clip, vf_loss_scale,
                 gamma=0.99, gae_lambda=0.95, normalize_gae=True,
                 policy_reg_scale=0, lr=3e-4):
        super().__init__()

        self.policy = policy
        self.fixed_policy = fixed_policy
        self.optim = torch.optim.Adam(policy.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()

        self.clip = clip
        self.vf_loss_scale = vf_loss_scale

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_gae = normalize_gae

        self.policy_reg_scale = policy_reg_scale

    def to(self, device):
        self.policy.to(device)
        self.fixed_policy.to(device)
        return super().to(device)

    def prepare_batch(self, episodes, max_context, context_l, process_batch_size):
        full_batch = BatchEpisode(episodes, max_context, context_l)

        # compute initial log probs & advantages
        with torch.cuda.amp.autocast(), torch.no_grad():
            if process_batch_size == None:
                raise
                init_log_probs, vs = self.policy.dist_with_v(
                    full_batch.states
                ).log_prob(full_batch.actions)
            else:
                all_init_log_probs_chunks = []
                all_vs_chunks = []
                all_actions_chunks = {
                    k: v.split(process_batch_size)
                    for k, v in full_batch.all_actions.items()
                }
                all_action_masks_chunks = {
                    k: v.split(process_batch_size)
                    for k, v in full_batch.all_action_masks.items()
                }
                for chunk_i, (all_states_chunk, all_firsts_chunk, all_last_indices_chunk) in enumerate(zip(
                    full_batch.all_stacked_states.split(process_batch_size),
                    full_batch.all_firsts.split(process_batch_size),
                    full_batch.all_last_indices.split(process_batch_size)
                )):
                    all_states_chunk = all_states_chunk.to(self.device)
                    all_firsts_chunk = all_firsts_chunk.to(self.device)
                    all_last_indices_chunk = all_last_indices_chunk.to(self.device)
                    
                    all_actions_chunk = {
                        k: v[chunk_i].to(self.device)
                        for k, v in all_actions_chunks.items()
                    }
                    all_action_masks_chunk = {
                        k: v[chunk_i].to(self.device)
                        for k, v in all_action_masks_chunks.items()
                    }

                    all_dists, all_vs = self.policy.dist_with_v(all_states_chunk, all_action_masks_chunk, all_firsts_chunk, all_last_indices_chunk)
                    
                    all_hierarchy_masks_chunk = self.policy.process_action_hierarchy_mask(all_actions_chunk, all_action_masks_chunk)
                    all_hierarchy_masks_chunk = {k: v.to(self.device) for k, v in all_hierarchy_masks_chunk.items()}
                    all_init_log_probs = all_dists.log_prob(all_actions_chunk, all_hierarchy_masks_chunk)
    
                    all_init_log_probs_chunks.append(all_init_log_probs)
                    all_vs_chunks.append(all_vs)

                init_log_probs = torch.cat(all_init_log_probs_chunks, dim=0)[full_batch.curr_mask].cpu()
                all_vs = torch.cat(all_vs_chunks, dim=0).cpu()
            
            full_batch.init_log_probs = init_log_probs
            full_batch.compute_gae(all_vs, self.gamma, self.gae_lambda, process_batch_size)

        return full_batch


    def step(self, episodes, max_context, context_l, n_max_epoch, batch_size, target_kl=None, process_batch_size=None):
        full_batch = self.prepare_batch(episodes, max_context, context_l, process_batch_size)
        loader = torch_data.DataLoader(
            full_batch, batch_size=None,
            sampler=torch_data.BatchSampler(
                torch_data.RandomSampler(full_batch), batch_size=batch_size, drop_last=False
            )
        )
        step_i = 0
        self.optim.zero_grad(set_to_none=True)
        for _ in range(n_max_epoch):
            for batch in loader:
                for k, v in batch.__dict__.items():
                    if type(v) == dict:
                        batch.__dict__[k] = {kk: vv.to(self.device) for kk, vv in v.items()}
                    else:
                        batch.__dict__[k] = v.to(self.device)

                with torch.cuda.amp.autocast():
                    step_i += 1

                    # optimize this
                    dists, vs = self.policy.dist_with_v(batch.states, batch.action_masks, batch.firsts, batch.last_indices)
                    with torch.no_grad():
                        fixed_dists = self.fixed_policy.dist(batch.states, batch.action_masks, batch.firsts, batch.last_indices)
                    del batch.states

                    v_target = batch.gaes + batch.values
                    vf_loss = (vs - v_target).pow(2).mean(0)

                    if self.normalize_gae:
                        advs = (batch.gaes - batch.gaes.mean()) / (batch.gaes.std() + 1e-6)
                    else:
                        advs = batch.gaes

                    hierarchy_masks = self.policy.process_action_hierarchy_mask(batch.actions, batch.action_masks)
                    hierarchy_masks = {k: v.to(self.device) for k, v in hierarchy_masks.items()}
                    log_probs = dists.log_prob(batch.actions, hierarchy_masks)
                    log_ratios = log_probs - batch.init_log_probs
                    ratios = log_ratios.exp()

                    # clipped surrogate loss
                    policy_loss_1 = advs * ratios
                    policy_loss_2 = advs * ratios.clamp(1 - self.clip, 1 + self.clip)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean(0)

                    kls = []
                    for k in dists.categorical_dict:
                        if k != 'craft_items':
                            kl = torch_dist.kl_divergence(
                                dists.categorical_dict[k],
                                fixed_dists.categorical_dict[k]
                            ).mean(0)
                            kls.append(kl)
                        else:
                            if hierarchy_masks['craft_items'].sum() > 0:
                                uniform = torch_dist.Categorical(
                                    probs=batch.action_masks['craft_items'][hierarchy_masks['craft_items']]
                                )
                                masked_dict = torch_dist.Categorical(
                                    probs=dists.categorical_dict['craft_items'].probs[hierarchy_masks['craft_items']]
                                )
                                kl = torch_dist.kl_divergence(masked_dict, uniform).mean(0)
                                kls.append(kl)
                    policy_reg_loss = torch.stack(kls).mean(0)

                    del fixed_dists

                self.scaler.scale(policy_loss).backward(retain_graph=True)
                self.scaler.scale(self.vf_loss_scale*vf_loss).backward(retain_graph=True)
                self.scaler.scale(self.policy_reg_scale*policy_reg_loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)

                with torch.no_grad():
                    approx_kl_div = ((ratios - 1) - log_ratios).mean(0).cpu().numpy()
                diverged = target_kl is not None and approx_kl_div > 1.5*target_kl
                if diverged:
                    break
            
            if diverged:
                break
        
        return {
            'vf_loss': vf_loss,
            'policy_loss': policy_loss,
            'policy_reg_loss': policy_reg_loss,
            'mean_max_prob': {
                k: dists.probs.max(-1).values.mean(0)
                for k, dists in dists.categorical_dict.items()
            },
            'mean_min_prob': {
                k: dists.probs.min(-1).values.mean(0)
                for k, dists in dists.categorical_dict.items()
            },
            'mean_ratio': ratios.mean(0),
            'min_v': vs.min(),
            'max_v': vs.max(),
            'mean_v': vs.mean(0),
            'approx_kl_div': approx_kl_div,
            'n_step': step_i
        }
