import ns_gym as nsg
import torch

def observation_type_checker(obs):
    if isinstance(obs, dict) and 'state' in obs:
        obs = obs['state']
    return obs

def reward_type_checker(reward):
    if isinstance(reward, nsg.base.Reward):
        reward = reward.reward
    return reward

def nn_model_input_checker(x):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    return x
