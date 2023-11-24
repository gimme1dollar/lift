import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.utils.data as torch_data

from simpl.nn import ToDeviceMixin

from .ppo import BatchEpisode

class MTBatchEpisode(BatchEpisode):
    def __init__(self, episodes, max_context, context_l):
        super().__init__(episodes, max_context, context_l)
        
        self.task_embeds = torch.cat([
            episode.task_embed[None, :].expand(len(episode), -1) for episode in episodes
        ])
        self.all_task_embeds = torch.cat([
            episode.task_embed[None, :].expand(len(episode)+1, -1) for episode in episodes
        ])
    
    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        ret.task_embeds = self.task_embeds[idx]
        return ret
    

class MTPPO(ToDeviceMixin, nn.Module):
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
        full_batch = MTBatchEpisode(episodes, max_context, context_l)
        
        # compute initial log probs & advantagnes
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
                for chunk_i, (all_states_chunk, all_firsts_chunk, all_last_indices_chunk, all_task_embeds_chunk) in enumerate(zip(
                    full_batch.all_stacked_states.split(process_batch_size),
                    full_batch.all_firsts.split(process_batch_size),
                    full_batch.all_last_indices.split(process_batch_size),
                    full_batch.all_task_embeds.split(process_batch_size)
                )):
                    all_states_chunk = all_states_chunk.to(self.device)
                    all_firsts_chunk = all_firsts_chunk.to(self.device)
                    all_last_indices_chunk = all_last_indices_chunk.to(self.device)
                    all_task_embeds_chunk = all_task_embeds_chunk.to(self.device)
                    
                    all_actions_chunk = {
                        k: v[chunk_i].to(self.device)
                        for k, v in all_actions_chunks.items()
                    }
                    all_action_masks_chunk = {
                        k: v[chunk_i].to(self.device)
                        for k, v in all_action_masks_chunks.items()
                    }
                    
                    all_dists, all_vs = self.policy.dist_with_v(all_states_chunk, all_action_masks_chunk, all_firsts_chunk, all_last_indices_chunk, all_task_embeds_chunk)
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
                torch_data.RandomSampler(full_batch), batch_size=batch_size, drop_last=True
            ),
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

                with torch.cuda.amp.autocast(cache_enabled=False):
                    step_i += 1

                    # optimize this
                    dists, vs = self.policy.dist_with_v(batch.states, batch.action_masks, batch.firsts, batch.last_indices, batch.task_embeds)
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

                    loss = policy_loss + self.vf_loss_scale*vf_loss + self.policy_reg_scale*policy_reg_loss

                self.scaler.scale(loss).backward()
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
            'loss': loss,
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