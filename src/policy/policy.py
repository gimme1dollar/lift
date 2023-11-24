import minedojo
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn.functional as F
from simpl.rl.policy import StochasticNNPolicy
from vpt.agent import resize_image, AGENT_RESOLUTION
from minedojo.sim.mc_meta.mc import ALL_CRAFT_SMELT_ITEMS

class MultiCategorical:
    def __init__(self, categorical_dict):
        self.categorical_dict = categorical_dict

    def sample(self, *args, **kwargs):
        return {
            k: dist.sample(*args, **kwargs)
            for k, dist in self.categorical_dict.items()
        }

    def rsample(self, *args, **kwargs):
        return {
            k: dist.rsample(*args, **kwargs)
            for k, dist in self.categorical_dict.items()
        }

    def log_prob(self, samples, hierarchy_masks=None):
        log_probs = []
        for k, dist in self.categorical_dict.items():
            if hierarchy_masks is None:
                log_probs.append(dist.log_prob(samples[k]))
            else:
                log_probs.append(
                    dist.log_prob(samples[k]) * hierarchy_masks[k]
                )
        return torch.stack(log_probs).sum(0)

    def argmax(self):
        return {
            k: dist.logits.argmax(-1)
            for k, dist in self.categorical_dict.items()
        }


@torch_dist.kl.register_kl(MultiCategorical, MultiCategorical)
def kl_multicat_multicat(p, q):
    return sum([
        torch_dist.kl_divergence(p_cat, q.categorical_dict[k])
        for k, p_cat in p.categorical_dict.items()
    ])


class VPTPolicy(StochasticNNPolicy):
    def __init__(self, minerl_agent, event_level_control=True):
        super().__init__()
        self.minerl_agent = minerl_agent
        self.vpt_policy = minerl_agent.policy  # to register module

        self.context_l = self.minerl_agent.context_l
        self.n_context_repeat = self.minerl_agent.n_context_repeat

        # vpt to MineDojo action space conversion
        sim = minedojo.sim.MineDojoSim(image_size=[10, 10], event_level_control=event_level_control)
        self.actionables = sim._sim_spec.actionables

    def to(self, device):
        self.minerl_agent.device = device
        self.vpt_policy.net.device = device
        self.minerl_agent._dummy_first = self.minerl_agent._dummy_first.to(device)
        return super().to(device)

    def new_context(self):
        return self.minerl_agent.policy.initial_state(1)


    def forward_net(self, batch_seq_processed_state, batch_first, batch_context):
        x = batch_seq_processed_state
        if self.minerl_agent.policy.net.diff_obs_process:
            processed_obs = self.minerl_agent.policy.net.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

        if self.minerl_agent.policy.net.pre_lstm_ln is not None:
            x = self.minerl_agent.policy.pre_lstm_ln(x)

        if self.minerl_agent.policy.net.recurrent_layer is not None:
            x, batch_context = self.minerl_agent.policy.net.recurrent_layer(
                x, batch_first[:, None].expand(-1, x.shape[1]), batch_context
            )
        else:
            raise

        x = F.relu(x, inplace=False)

        x = self.minerl_agent.policy.net.lastlayer(x)
        x = self.minerl_agent.policy.net.final_ln(x)

        return x, batch_context

    def forward_recurrent(self, batch_seq_processed_state, batch_firsts, context_l, n_context):
        batch_context = self.minerl_agent.policy.initial_state(len(batch_seq_processed_state))

        # torch.no_grad() has a bug when it is used with amp.autocat
        # https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475
        requires_grads = []
        for param in self.parameters():
            requires_grads.append(param.requires_grad)
            param.requires_grad_(False)
        for context_i in range(n_context-1):
            _, batch_context = self.forward_net(
                batch_seq_processed_state[:, context_i*context_l:(context_i+1)*context_l, ...],
                batch_firsts[:, context_i], batch_context
            )
        for param, requires_grad in zip(self.parameters(), requires_grads):
            param.requires_grad_(requires_grad)
        batch_seq_h, _ = self.forward_net(
            batch_seq_processed_state[:, -context_l:, ...],
            batch_firsts[:, -1], batch_context
        )
        return batch_seq_h

    def dist_with_v(self, batch_seq_processed_state, batch_action_mask, batch_firsts, batch_last_idx):
        context_l = self.context_l
        n_context = self.n_context_repeat

        batch_seq_h = self.forward_recurrent(batch_seq_processed_state, batch_firsts, context_l, n_context)
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

    def dist(self, batch_seq_processed_state, batch_action_mask, batch_firsts, batch_last_idx):
        context_l = self.context_l
        n_context = self.n_context_repeat

        batch_seq_h = self.forward_recurrent(batch_seq_processed_state, batch_firsts, context_l, n_context)
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

    def rollout_forward_net(self, processed_state, context):
        x = processed_state[None, None, ...]

        if self.minerl_agent.policy.net.diff_obs_process:
            processed_obs = self.minerl_agent.policy.net.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

        if self.minerl_agent.policy.net.pre_lstm_ln is not None:
            x = self.minerl_agent.policy.pre_lstm_ln(x)

        if self.minerl_agent.policy.net.recurrent_layer is not None:
            first = torch.tensor([False])[None, :].expand(len(x), 1).to(self.device)
            x, context = self.minerl_agent.policy.net.recurrent_layer(
                x, first, context
            )
        else:
            raise
            #state_out = state_in

        x = F.relu(x, inplace=False)

        x = self.minerl_agent.policy.net.lastlayer(x)
        x = self.minerl_agent.policy.net.final_ln(x)

        return x[0, 0], context

    def dist_action(self, processed_state, action_mask, context):
        batch_h, context = self.rollout_forward_net(processed_state, context)

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
            "buttons": action["buttons"].cpu().numpy()[None],  # they want this useless dims
            "camera": action["camera"].cpu().numpy()[None],
            "craft_items": action["craft_items"].cpu().numpy()[None],
        }
        minerl_action = self.minerl_agent.action_mapper.to_factored(action)
        minerl_action_transformed = self.minerl_agent.action_transformer.policy2env(minerl_action)
        
        # minerl -> minedojo
        for handler in self.actionables:
            if handler.to_string() not in minerl_action_transformed:
                minerl_action_transformed[handler.to_string()] = handler.space.no_op()
        
        # if action is to craft
        if minerl_action_transformed["inventory"] == 1: 
            craft_idx = action["craft_items"].item()
            minerl_action_transformed["craft"] = ALL_CRAFT_SMELT_ITEMS[craft_idx]
            
        return minerl_action_transformed

    def act(self, processed_state, action_mask, context):
        if self.explore is None:
            raise RuntimeError('explore is not set')

        processed_state = processed_state.to(self.device)
        action_mask = {
            k: v.to(self.device)
            for k, v in action_mask.items()
        }

        with torch.no_grad():
            training = self.training
            self.eval()
            dist, context = self.dist_action(processed_state, action_mask, context)
            self.train(training)

        if self.explore is True:
            batch_action = dist.sample()
        else:
            batch_action = dist.argmax()

        action = {
            "buttons": batch_action["buttons"],
            "camera": batch_action["camera"],
            "craft_items": batch_action["craft_items"],
        }
        return action, context
