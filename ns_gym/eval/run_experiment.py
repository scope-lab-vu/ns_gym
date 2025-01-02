import ns_gym
import gymnasium as gym
from ns_gym.utils import parse_config


"""
Generic experiment runner functions
- Reads a YAML config file or command line arguments
- Builds non-stationary env
- Configures benchmark algorithms
- Stores results csv, trace visualizer
"""


def make_env(config):
    pass

def run_episode(env,agent,seed,config):
    """Run an episode with a given agent and environment.
    """
    done = False
    truncated = False
    obs,_ = env.reset(seed)
    
    episode_reward = []
    
    while not done and not truncated:
        action = agent.act(obs)
        obs, reward, done, truncated = env.step(action)
        obs,reward = ns_gym.utils.type_mismatch_checker(obs,reward)
        episode_reward.append(reward)

    return sum(episode_reward), episode_reward


if __name__ == "__main__":
    pass