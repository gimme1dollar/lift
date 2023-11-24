import gym
import minedojo
import numpy as np

from omegaconf import OmegaConf

from minedojo.sim.wrappers.ar_nn.ar_masks_wrapper import ARMasksWrapper
from minedojo.tasks import _resource_file_path, _parse_inventory_dict, MetaTaskName2Class

eval_task_ids = [
    'harvest_milk_with_empty_bucket_and_cow',
    'harvest_wool_with_shears_and_sheep',
    'harvest_1_chicken',
    'harvest_1_log',
    'combat_cow_plains_diamond_armors_diamond_sword_shield',
    'combat_sheep_plains_diamond_armors_diamond_sword_shield',
    'combat_spider_plains_diamond_armors_diamond_sword_shield',
    'combat_zombie_plains_diamond_armors_diamond_sword_shield',
]

eval_task_specs = {
    eval_task_id: minedojo.tasks.ALL_TASKS_SPECS[eval_task_id].copy()
    for eval_task_id in eval_task_ids
}

task_random = np.random.RandomState(seed=42)
world_seeds = task_random.randint(99999, size=len(eval_task_ids))
seeds = task_random.randint(99999, size=len(eval_task_ids))

for eval_task_id, seed, world_seed in zip(eval_task_ids, seeds, world_seeds):
    spec = eval_task_specs[eval_task_id]
    
    spec['image_size'] = [360, 640]
    
    spec['use_voxel'] = False
    spec['use_lidar'] = False
    spec['fast_reset'] = False
    spec['fast_reset_random_teleport_range'] = 0
    
    spec['seed'] = int(seed)
    spec['world_seed'] = int(world_seed)
    
    if eval_task_id == 'harvest_1_chicken':
        spec['initial_mobs'] = 'chicken'
    
    if 'initial_mobs' in spec:
        spec['initial_mob_spawn_range_low'] = [-4, 1, 5]
        spec['initial_mob_spawn_range_high'] = [4, 1, 8]
    
    if eval_task_id == 'harvest_1_log':
        spec['specified_biome']  = 'forest'
    else:
        spec['specified_biome']  = 'sunflower_plains'
        
    del spec['prompt']


def make_eval_env(task_spec):
    task_spec = dict(task_spec)
    if "initial_inventory" in task_spec:
        task_spec["initial_inventory"] = _parse_inventory_dict(task_spec["initial_inventory"])
    
    meta_task_cls = task_spec.pop('__cls__').lower()
    env = MetaTaskName2Class[meta_task_cls](**task_spec)
    env = ARMasksWrapper(env)
    
    return env
