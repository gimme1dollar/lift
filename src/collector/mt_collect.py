import time
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from simpl.collector import BaseWorker
from minedojo.sim.wrappers import FastResetWrapper


class MTEpisode:
    def __init__(self, init_state, init_processed_state, action_mask, info, task_embed):
        self.states = [init_state]
        self.processed_states = [init_processed_state]
        self.actions = []
        self.action_masks = [action_mask]
        self.rewards = []
        self.dones = []
        self.infos = []
        
        self.task_embed = task_embed

    def __repr__(self):
        return f'MTEpisode(cum_reward={sum(self.rewards)}, length={len(self)})'

    def __len__(self):
        return len(self.actions)
    
    def add_step(self, action, next_state, next_processed_state, action_mask, reward, done, info):
        self.actions.append(action)
        self.states.append(next_state)
        self.processed_states.append(next_processed_state)
        self.action_masks.append(action_mask)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        

class MTCollector:
    def __init__(self, env, time_limit=None, n_initial_step=10):
        self.env = env
        self.time_limit = time_limit if time_limit is not None else np.inf
        
        self.n_initial_step = n_initial_step

    def collect_episode(self, actor, env_instance, world_seed, task_embed):
        if env_instance is None:
            assert len(self.env.unwrapped._sim_spec._world_generator_handlers) == 1
            self.env.unwrapped._sim_spec._world_generator_handlers[0].world_seed = world_seed
            env = self.env
        else:
            env = env_instance
            env.unwrapped._bridge_env = self.env.unwrapped._bridge_env
        
        state, done, t = env.reset(), False, 0

        for _ in range(self.n_initial_step):
            action = env.action_space.no_op()
            state, *_ = env.step(action)

        info = {}
        processed_state = actor.process_obs(state)
        action_mask = actor.process_action_mask(state)
        context = actor.new_context()
        episode = MTEpisode(state, processed_state, action_mask, info, task_embed)
        self.episode = episode # for debug...

        while not done and t < self.time_limit:
            if task_embed is not None:
                action, context = actor.act(processed_state, action_mask, context, task_embed)
            else:
                action, context = actor.act(processed_state, action_mask, context)
            env_action = actor.process_action(action)
            state, reward, done, info = env.step(env_action)
            
            processed_state = actor.process_obs(state)
            action_mask = actor.process_action_mask(state)
            data_done = done
            if 'TimeLimit.truncated' in info:
                data_done = not info['TimeLimit.truncated']
            episode.add_step(action, state, processed_state, action_mask, reward, data_done, info)
            t += 1
            
        if env_instance is not None:
            env_instance.unwrapped._bridge_env = None

        return episode
