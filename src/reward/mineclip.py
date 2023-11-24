import numpy as np

import torch
import torch.nn as nn

import mineclip

cfg = {
    'arch': 'vit_base_p16_fz.v2.t2',
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': 'attn.d2.nh8.glusw',
    'resolution': [160, 256]
}
n_stack = 16
hidden_dim = cfg['hidden_dim']


class MineCLIP(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        
        mineclip_model = mineclip.MineCLIP(**cfg)
        mineclip_model.load_ckpt(model_path, strict=True)
        
        self.mineclip_model = mineclip_model
        
    def embed_text(self, texts):
        """ 
        list of python strings
        Not normalized yet
        tokenize_text + encode_text
        """
        text_embeds = self.mineclip_model.encode_text(texts)
        if self.mineclip_model.reward_head.text_residual_weight is None:
            text_embeds = self.mineclip_model.reward_head.text_adapter(text_embeds)
        else:
            res = torch.sigmoid(self.mineclip_model.reward_head.text_residual_weight)
            text_embeds = res * text_embeds + (1.0 - res) * self.mineclip_model.reward_head.text_adapter(text_embeds)
        return text_embeds
    
    def embed_video(self, seq_img, n_pad=n_stack-2, process_batch_size=None):
        """ 
        Don't support batch computing
        input : (T, 3, 256, 160) 0~255 image
        default n_pad produces T-1 embeds
        Not normalized yet
        preprocess + image_encode + video_encode
        """
        seq_img = mineclip.utils.basic_image_tensor_preprocess(
            seq_img,
            mineclip.mineclip.base.MC_IMAGE_MEAN, mineclip.mineclip.base.MC_IMAGE_STD
        )

        if process_batch_size is None:
            seq_img_feature = self.mineclip_model.image_encoder(seq_img)
        else:
            seq_img_feature = torch.cat([
                self.mineclip_model.image_encoder(seq_img_chunk)
                for seq_img_chunk in seq_img.split(process_batch_size)
            ])

        seq_img_feature = torch.cat([
            seq_img_feature[[0]].expand(n_pad, -1),
            seq_img_feature,
        ], dim=0)

        seq_stack_img_feature = []
        for i in range(len(seq_img)-1):
            seq_stack_img_feature.append(seq_img_feature[i:i+n_stack])
        seq_stack_img_feature = torch.stack(seq_stack_img_feature, dim=0)

        seq_video_embed = self.mineclip_model.forward_video_features(seq_stack_img_feature)

        if self.mineclip_model.reward_head.video_residual_weight is None:
            seq_video_embed = self.mineclip_model.reward_head.video_adapter(seq_video_embed)
        else:
            res = torch.sigmoid(self.mineclip_model.reward_head.video_residual_weight)
            seq_video_embed = res * seq_video_embed + (1.0 - res) * self.mineclip_model.reward_head.video_adapter(seq_video_embed)
        
        return seq_video_embed
    
    def compute_reward(self, task_embed, video_embed, temperature_scale=1., normalize_embed=True):
        temperature = temperature_scale / self.mineclip_model.clip_model.logit_scale.exp()
        
        if normalize_embed == True:
            task_embed /= task_embed.norm(p=2, dim=-1, keepdim=True)
            video_embed /= video_embed.norm(p=2, dim=-1, keepdim=True)
        
        return (video_embed @ task_embed / temperature).cpu().numpy()


def soften(rewards, window_size):
    return np.array([
        rewards[max(0, t+1-window_size):t+1].mean()
        for t in range(len(rewards))
    ])

def min_clip(rewards, min_value):
    return (rewards - min_value).clip(0, np.inf)

def zero_out_decreased(rewards):
    new_rewards = []
    best_r = -np.inf
    for r in rewards:
        if r > best_r:
            new_rewards.append(r)
            best_r = r
        else:
            new_rewards.append(0)
    return np.array(new_rewards)
