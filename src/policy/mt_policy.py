import minedojo
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist

from simpl.rl.policy import StochasticNNPolicy

from vpt.agent import resize_image, AGENT_RESOLUTION
from vpt.lib import torch_util as tu
from vpt.lib.masked_attention import get_mask
from vpt.lib.xf import attention, StridedAttn

from .policy import MultiCategorical
from .adapter import Adapter, NormalInitLinear


class MTAdapterResidualRecurrentBlocks(nn.Module):
    def __init__(self, module, task_embed_dim, adapter_dim_factor):
        super().__init__()
        self.module = module
        self.blocks = module.blocks
        
        for recurrent_block in self.module.blocks:
            in_dim = recurrent_block.r.orc_block.proj_layer.out_features
            z_dim = in_dim // adapter_dim_factor
            recurrent_block.r.orc_block = MTAdapterSelfAttentionLayer(
                recurrent_block.r.orc_block, in_dim, z_dim, task_embed_dim
            )
        
        for i, recurrent_block in enumerate(self.blocks):
            recurrent_block.r = MTMaskedAttention(recurrent_block.r)

        for i, recurrent_block in enumerate(self.blocks):
            in_dim = recurrent_block.mlp1.layer.out_features
            z_dim = in_dim // adapter_dim_factor
            self.blocks[i] = MTAdapterRecurrentBlock(
                recurrent_block, in_dim, z_dim, task_embed_dim
            )
    
    def forward(self, x, first, state, task_embed):
        state_out = []
        assert len(state) == len(
            self.blocks
        ), f"Length of state {len(state)} did not match length of blocks {len(self.blocks)}"
        for block, _s_in in zip(self.blocks, state):
            x, _s_o = block(x, first, _s_in, task_embed)
            state_out.append(_s_o)
        return x, state_out

    def initial_state(self, batchsize):
        return self.module.initial_state(batchsize)


def recurrent_forward(module, x, first, state, task_embed, reverse_lstm=False):
    if isinstance(module, nn.LSTM):
        raise NotImplementedError
    else:
        return module(x, first, state, task_embed)


class MTAdapter(nn.Module):
    def __init__(self, in_dim, z_dim, task_embed_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            NormalInitLinear(in_dim+task_embed_dim, z_dim),
            nn.ReLU(),
            NormalInitLinear(z_dim, in_dim),
        )
        
    def forward(self, x, t):
        return x + self.net(torch.cat([x, t], dim=-1))


class MTAdapterRecurrentBlock(nn.Module):
    def __init__(self, module, in_dim, z_dim, task_embed_dim):
        super().__init__()
        
        self.module = module
        self.r = self.module.r
        self.adapter = MTAdapter(in_dim, z_dim, task_embed_dim)
    
    def forward(self, x, first, state, task_embed):
        residual = x
        x = self.module.pre_r_ln(x)
        x, state_out = recurrent_forward(
            self.r,
            x,
            first,
            state,
            task_embed,
            reverse_lstm=self.module.recurrence_type == "multi_layer_bilstm" and (self.block.block_number + 1) % 2 == 0,
        )
        if self.module.is_residual and "lstm" in self.module.recurrence_type:  # Transformer already residual.
            x = x + residual
        if self.module.use_pointwise_layer:
            # Residual MLP
            residual = x
            x = self.module.mlp1(self.module.mlp0(x))
            x = self.adapter(x, task_embed)
            if self.module.is_residual:
                x = x + residual
        return x, state_out

    
    
class MTMaskedAttention(nn.Module):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        self.orc_attn = module.orc_attn
        self.orc_block = module.orc_block
        self.maxlen = module.maxlen
        
        self.mask = module.mask
        self.heads = module.heads

    def initial_state(self, batchsize: int, device=None):
        return self.module.initial_state(batchsize, device)

    def forward(self, input_bte, first_bt, state, task_embed):
        state_mask, xf_state = state
        t = first_bt.shape[1]
        if self.mask == "clipped_causal":
            new_mask, state_mask = get_mask(
                first_b11=first_bt[:, [[0]]],
                state_mask=state_mask,
                t=t,
                T=t + self.maxlen,
                maxlen=self.maxlen,
                heads=self.heads,
                device=input_bte.device,
            )
        else:
            raise
        output, xf_state = self.orc_block(input_bte, xf_state, new_mask, task_embed)

        return output, (state_mask, xf_state)

    def get_log_keys(self):
        # These are logged in xf.SelfAttentionLayer
        return [f"activation_{stat}/{self.log_scope}/{k}" for k in ["K", "Q", "V", "A", "Aproj"] for stat in ["mean", "std"]]

    
class MTAdapterSelfAttentionLayer(nn.Module):
    def __init__(self, module, in_dim, z_dim, task_embed_dim):
        super().__init__()
        
        self.module = module
        self.attn = module.attn
        self.relattn = module.relattn
        self.maxlen = module.maxlen
        self.dtype = module.dtype
        
        self.ln_x = module.ln_x
        self.q_layer = module.q_layer
        self.k_layer = module.k_layer
        self.v_layer = module.v_layer
        self.proj_layer = module.proj_layer
        
        self.use_muP_factor = module.use_muP_factor
        
        self.adapter = MTAdapter(in_dim, z_dim, task_embed_dim)
        
    def forward(self, X_bte, state, mask, task_embeding_bte):
        R_bte, state = self.residual(X_bte, state, mask)
        
        return X_bte + self.adapter(R_bte, task_embeding_bte), state

    def initial_state(self, batchsize, initial_T=0):
        return (
            tu.zeros((batchsize, initial_T, self.module.x_size), dtype=self.module.dtype),
            tu.zeros((batchsize, initial_T, self.module.x_size), dtype=self.module.dtype),
        )

    def residual(self, X_bte, state, mask):
        X_bte = self.ln_x(X_bte)
        Q_bte = self.q_layer(X_bte)
        K_bte = self.k_layer(X_bte)
        V_bte = self.v_layer(X_bte)
        if state:
            state, K_bte, V_bte = self.update_state(state, K_bte, V_bte)
        postproc_closure, Q_bte, K_bte, V_bte = self.attn.preproc_qkv(Q_bte, K_bte, V_bte)
        extra_btT = self.relattn_logits(X_bte, K_bte.shape[1]) if self.relattn else None
        A_bte = attention(
            Q_bte,
            K_bte,
            V_bte,
            mask=mask,
            extra_btT=extra_btT,
            maxlen=self.maxlen,
            dtype=self.dtype,
            check_sentinel=isinstance(self.attn, StridedAttn),
            use_muP_factor=self.use_muP_factor,
        )
        A_bte = postproc_closure(A_bte)
        Aproj_bte = self.proj_layer(A_bte)
        return Aproj_bte, state
    
    def update_state(self, state, K_bte, V_bte):
        return self.module.update_state(state, K_bte, V_bte)
    
    def relattn_logits(self, X_bte, T):
        return self.module.relattn_logits(X_bte, T)


class MTVPTPolicy(StochasticNNPolicy):
    def __init__(self, minerl_agent, task_embed_dim, adapter_factor, event_level_control=True):
        super().__init__()
        
        minerl_agent.policy.net.recurrent_layer = MTAdapterResidualRecurrentBlocks(
            minerl_agent.policy.net.recurrent_layer,
            task_embed_dim, adapter_factor
        )
        minerl_agent.policy.requires_grad_(False)
        for recurrent_block in minerl_agent.policy.net.recurrent_layer.blocks:
            recurrent_block.adapter.requires_grad_(True)
            recurrent_block.module.r.orc_block.adapter.requires_grad_(True)
        minerl_agent.policy.value_head.requires_grad_(True)
        
        self.minerl_agent = minerl_agent
        self.vpt_policy = minerl_agent.policy  # to register module
        
        sim = minedojo.sim.MineDojoSim(image_size=[10, 10], event_level_control=event_level_control)
        self.actionables = sim._sim_spec.actionables
    
    def to(self, device):
        self.minerl_agent.device = device
        self.vpt_policy.net.device = device
        self.minerl_agent._dummy_first = self.minerl_agent._dummy_first.to(device)
        return super().to(device)
    
    def new_context(self):
        return self.minerl_agent.policy.initial_state(1)

    def forward_net(self, batch_seq_processed_state, batch_first, batch_context, batch_task_embed):
        x = batch_seq_processed_state
        if self.minerl_agent.policy.net.diff_obs_process:
            processed_obs = self.minerl_agent.policy.net.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

        if self.minerl_agent.policy.net.pre_lstm_ln is not None:
            x = self.minerl_agent.policy.pre_lstm_ln(x)

        if self.minerl_agent.policy.net.recurrent_layer is not None:
            x, batch_context = self.minerl_agent.policy.net.recurrent_layer(
                x, batch_first[:, None].expand(-1, x.shape[1]),
                batch_context, batch_task_embed[:, None, :].expand(-1, x.shape[1], -1)
            )
        else:
            raise

        x = F.relu(x, inplace=False)

        x = self.minerl_agent.policy.net.lastlayer(x)
        x = self.minerl_agent.policy.net.final_ln(x)
        
        return x, batch_context
    
    def forward_recurrent(self, batch_seq_processed_state, batch_firsts, context_l, n_context, batch_task_embed):
        batch_context = self.minerl_agent.policy.initial_state(len(batch_seq_processed_state))
        for context_i in range(n_context-1):
            with torch.no_grad():
                _, batch_context = self.forward_net(
                    batch_seq_processed_state[:, context_i*context_l:(context_i+1)*context_l, ...],
                    batch_firsts[:, context_i], batch_context, batch_task_embed
                )
        batch_seq_h, _ = self.forward_net(
            batch_seq_processed_state[:, -context_l:, ...],
            batch_firsts[:, -1], batch_context, batch_task_embed
        )
        return batch_seq_h
    
    def dist_with_v(self, batch_seq_processed_state, batch_action_mask, batch_firsts, batch_last_idx, batch_task_embed):
        context_l = self.minerl_agent.policy.net.recurrent_layer.blocks[0].r.maxlen
        n_context = batch_seq_processed_state.shape[1] // context_l
          
        batch_seq_h = self.forward_recurrent(batch_seq_processed_state, batch_firsts, context_l, n_context, batch_task_embed)
        batch_h = batch_seq_h.gather(1, batch_last_idx[:, None, None].expand(-1, -1, batch_seq_h.shape[-1]))[:, 0, :]
        
        logits_dict = self.minerl_agent.policy.pi_head(batch_h)
        categorical_dict = {}
        for k, logits in logits_dict.items():
            if k in batch_action_mask:
                zero_out_mask = (~batch_action_mask[k]).float() * -1e9
                logits = logits + zero_out_mask.unsqueeze(1)
            categorical_dict[k] = torch_dist.Categorical(logits=logits.squeeze(1))
        batch_dist = MultiCategorical(categorical_dict)
                                      
        batch_v = self.minerl_agent.policy.value_head(batch_h).squeeze(-1)

        return batch_dist, batch_v

    def dist(self, batch_seq_processed_state, batch_action_mask, batch_firsts, batch_last_idx, batch_task_embed):
        context_l = self.minerl_agent.policy.net.recurrent_layer.blocks[0].r.maxlen
        n_context = batch_seq_processed_state.shape[1] // context_l
            
        batch_seq_h = self.forward_recurrent(batch_seq_processed_state, batch_firsts, context_l, n_context, batch_task_embed)
        batch_h = batch_seq_h.gather(1, batch_last_idx[:, None, None].expand(-1, -1, batch_seq_h.shape[-1]))[:, 0, :]
        
        logits_dict = self.minerl_agent.policy.pi_head(batch_h)
        categorical_dict = {}
        for k, logits in logits_dict.items():
            if k in batch_action_mask:
                zero_out_mask = (~batch_action_mask[k]).float() * -1e9
                logits = logits + zero_out_mask.unsqueeze(1)
            categorical_dict[k] = torch_dist.Categorical(logits=logits.squeeze(1))
        batch_dist = MultiCategorical(categorical_dict)

        return batch_dist
    
    def rollout_forward_net(self, processed_state, context, task_embed):
        x = processed_state[None, None, ...]
        task_embed = task_embed[None, None, ...]
        
        if self.minerl_agent.policy.net.diff_obs_process:
            processed_obs = self.minerl_agent.policy.net.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

        if self.minerl_agent.policy.net.pre_lstm_ln is not None:
            x = self.minerl_agent.policy.pre_lstm_ln(x)

        if self.minerl_agent.policy.net.recurrent_layer is not None:
            first = torch.tensor([False])[None, :].expand(len(x), 1).to(self.device)
            x, context = self.minerl_agent.policy.net.recurrent_layer(
                x, first, context, task_embed
            )
        else:
            raise
            #state_out = state_in

        x = F.relu(x, inplace=False)

        x = self.minerl_agent.policy.net.lastlayer(x)
        x = self.minerl_agent.policy.net.final_ln(x)
        
        return x[0, 0], context

    def dist_action(self, processed_state, action_mask, context, task_embed):
        batch_h, context = self.rollout_forward_net(processed_state, context, task_embed)
        
        logits_dict = self.minerl_agent.policy.pi_head(batch_h)
        
        categorical_dict = {}
        for k, logits in logits_dict.items():
            # believe training will correctly handle if there's no possible action
            if k in action_mask and action_mask[k].any():
                logits[~action_mask[k].unsqueeze(0)] = -torch.inf
            categorical_dict[k] = torch_dist.Categorical(logits=logits.squeeze(0))
        batch_dist = MultiCategorical(categorical_dict)

        return batch_dist, context
    
    def process_obs(self, minerl_obs):
        minerl_obs["rgb"] = np.rollaxis(minerl_obs["rgb"], 0, 3)
        agent_input = resize_image(minerl_obs["rgb"], AGENT_RESOLUTION)
        agent_input = torch.from_numpy(agent_input).to(self.device)[None, None, :]
        
        with torch.no_grad():  # caution
            agent_input = self.minerl_agent.policy.net.img_preprocess(agent_input)
            agent_input = self.minerl_agent.policy.net.img_process(agent_input)
        
        return agent_input[0, 0, ...].cpu()

    def process_action_mask(self, obs):
        return {
            'craft_items': torch.as_tensor(obs['masks']['craft_smelt']),
        }

    def process_action_hierarchy_mask(self, batch_action, batch_action_mask):
        return {
            'buttons': torch.ones(len(batch_action['buttons']), dtype=bool),
            'camera': torch.as_tensor(~ self.minerl_agent.action_mapper.BUTTON_IDX_TO_CAMERA_META_OFF[batch_action['buttons'].cpu().numpy()]),
            'craft_items': batch_action_mask['craft_items'].any(-1), ## TODO : if inventory is full?
        }

    def process_action(self, action):
        action = {
            k: v.cpu().numpy()[None]  # they want this useless dims
            for k, v in action.items()
        }
        minerl_action = self.minerl_agent.action_mapper.to_factored(action)
        minerl_action_transformed = self.minerl_agent.action_transformer.policy2env(minerl_action)
        
        # minerl -> minedojo
        for handler in self.actionables:
            if handler.to_string() not in minerl_action_transformed:
                minerl_action_transformed[handler.to_string()] = handler.space.no_op()
        
        return minerl_action_transformed

    def act(self, processed_state, action_mask, context, task_embed):
        if self.explore is None:
            raise RuntimeError('explore is not set')

        processed_state = processed_state.to(self.device)
        task_embed = task_embed.to(self.device)

        with torch.no_grad():
            training = self.training
            self.eval()
            dist, context = self.dist_action(processed_state, action_mask, context, task_embed)
            self.train(training)

        if self.explore is True:
            action = dist.sample()
        else:
            action = dist.argmax()

        return action, context
