import pickle
from vpt.agent import MineRLAgent

def load(model_path, weights_path, context_l=None):
    agent_parameters = pickle.load(open(model_path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    agent = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, device='cpu')
    agent.load_weights(weights_path)

    if context_l is not None:
        for block in agent.policy.net.recurrent_layer.blocks:
            block.r.maxlen = context_l
            block.r.orc_block.maxlen = context_l
            block.r.orc_block.cache_keep_len = context_l
    else:
        context_l = agent.policy.net.recurrent_layer.blocks[0].r.maxlen
    agent.context_l = context_l
    agent.n_context_repeat = len(agent.policy.net.recurrent_layer.blocks)  # transformer-XL : the effect of context caching is proportional to the number of layers
    
    return agent

