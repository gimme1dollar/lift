import torch
import torch.nn as nn

from vpt.lib import torch_util as tu
from vpt.lib.util import recurrent_forward


init_std = 1e-3


class NormalInitLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.weight, std=init_std,
            a=-2*init_std, b=2*init_std
        )
        if self.bias is not None:
            nn.init.trunc_normal_(
                self.bias, std=init_std,
                a=-2*init_std, b=2*init_std
            )

class Adapter(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            NormalInitLinear(in_dim, z_dim),
            nn.ReLU(),
            NormalInitLinear(z_dim, in_dim),
        )
        
    def forward(self, x):
        return x + self.net(x)

class AdapterSelfAttentionLayer(nn.Module):
    def __init__(self, module, in_dim, z_dim):
        super().__init__()
        
        self.module = module
        self.attn = self.module.attn
        self.adapter = Adapter(in_dim, z_dim)
        
    def forward(self, X_bte, state, mask):
        R_bte, state = self.module.residual(X_bte, state, mask)
        
        return X_bte + self.adapter(R_bte), state

    def initial_state(self, batchsize, initial_T=0):
        return (
            tu.zeros((batchsize, initial_T, self.module.x_size), dtype=self.module.dtype),
            tu.zeros((batchsize, initial_T, self.module.x_size), dtype=self.module.dtype),
        )


class AdapterRecurrentBlock(nn.Module):
    def __init__(self, module, in_dim, z_dim):
        super().__init__()
        
        self.module = module
        self.r = self.module.r
        self.adapter = Adapter(in_dim, z_dim)
    
    def forward(self, x, first, state):
        residual = x
        x = self.module.pre_r_ln(x)
        x, state_out = recurrent_forward(
            self.module.r,
            x,
            first,
            state,
            reverse_lstm=self.module.recurrence_type == "multi_layer_bilstm" and (self.block.block_number + 1) % 2 == 0,
        )
        if self.module.is_residual and "lstm" in self.module.recurrence_type:  # Transformer already residual.
            x = x + residual
        if self.module.use_pointwise_layer:
            # Residual MLP
            residual = x
            x = self.module.mlp1(self.module.mlp0(x))
            x = self.adapter(x)
            if self.module.is_residual:
                x = x + residual
        return x, state_out


def add_adapters(vpt_policy, dim_factor):
    p = vpt_policy.minerl_agent.policy
    for recurrent_block in p.net.recurrent_layer.blocks:
        in_dim = recurrent_block.r.orc_block.proj_layer.out_features
        z_dim = in_dim // dim_factor
        recurrent_block.r.orc_block = AdapterSelfAttentionLayer(
            recurrent_block.r.orc_block, in_dim, z_dim
        )

    for i, recurrent_block in enumerate(p.net.recurrent_layer.blocks):
        in_dim = recurrent_block.mlp1.layer.out_features
        z_dim = in_dim // dim_factor
        p.net.recurrent_layer.blocks[i] = AdapterRecurrentBlock(
            recurrent_block, in_dim, z_dim
        )

    p.requires_grad_(False)
    for recurrent_block in p.net.recurrent_layer.blocks:
        recurrent_block.adapter.requires_grad_(True)
        recurrent_block.module.r.orc_block.adapter.requires_grad_(True)
    p.value_head.requires_grad_(True)
    p.pi_head.craft_items.requires_grad_(True)
