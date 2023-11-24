

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

    def step(self, episodes, max_context, context_l, n_max_epoch, batch_size, target_kl=None, process_batch_size=None):
        full_batch = BatchEpisode(episodes, max_context, context_l)
        
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
                    
                    all_dists, all_vs = self.policy.dist_with_v(all_states_chunk, all_firsts_chunk, all_last_indices_chunk)
                    all_init_log_probs = all_dists.log_prob(all_actions_chunk)
    
                    all_init_log_probs_chunks.append(all_init_log_probs)
                    all_vs_chunks.append(all_vs)

                init_log_probs = torch.cat(all_init_log_probs_chunks, dim=0)[full_batch.curr_mask].cpu()
                all_vs = torch.cat(all_vs_chunks, dim=0).cpu()
            
            full_batch.init_log_probs = init_log_probs
            full_batch.compute_gae(all_vs, self.gamma, self.gae_lambda, process_batch_size)
        
        # ppo epoch
        loader = torch_data.DataLoader(
            full_batch, batch_size=None,
            sampler=torch_data.BatchSampler(
                torch_data.RandomSampler(full_batch), batch_size=batch_size, drop_last=True
            ),
        )
        step_i = 0
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
                    dists, vs = self.policy.dist_with_v(batch.states, batch.firsts, batch.last_indices)

                    v_target = batch.gaes + batch.values
                    vf_loss = (vs - v_target).pow(2).mean(0)

                    if self.normalize_gae:
                        advs = (batch.gaes - batch.gaes.mean()) / batch.gaes.std()
                    else:
                        advs = batch.gaes

                    log_probs = dists.log_prob(batch.actions)
                    log_ratios = log_probs - batch.init_log_probs
                    ratios = log_ratios.exp()

                    # clipped surrogate loss
                    policy_loss_1 = advs * ratios
                    policy_loss_2 = advs * ratios.clamp(1 - self.clip, 1 + self.clip)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean(0)

                    with torch.no_grad():
                        fixed_dists = self.fixed_policy.dist(batch.states, batch.firsts, batch.last_indices)
                    policy_reg_loss = torch_dist.kl_divergence(dists, fixed_dists).mean(0)

                    loss = policy_loss + self.vf_loss_scale*vf_loss + self.policy_reg_scale*policy_reg_loss

                self.optim.zero_grad(set_to_none=True)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
        
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
