import ns_gym
import gymnasium as gym
from ns_gym.utils import parse_config
import multiprocessing

"""
Generic experiment runner
- Reads a YAML config file or command line arguments
- Builds non-stationary env
- Configures benchmark algorithms
- Stores results csv, trace visualizer
"""


def make_env(config):
    pass

def run_episode(config):
    pass

def main(config):
    pass

if __name__ == "__main__":
    config = parse_config()