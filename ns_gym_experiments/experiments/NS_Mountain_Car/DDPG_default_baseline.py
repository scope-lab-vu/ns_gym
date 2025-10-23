import ns_gym.base
from ns_gym.eval import run_experiment
from ns_gym.utils import parse_config   
from ns_gym.benchmark_algorithms import PPO,PPOActor, PPOCritic
import gymnasium as gym
import numpy as np

import torch

from stable_baselines3 import DDPG

def make_env(config):
    env = gym.make(config["env_name"])
    return env

def make_agent(config):

    model_path = config["model_path"]
    ddpg_agent = DDPG.load(model_path)
    wrapped_agent = ns_gym.base.StableBaselineWrapper(ddpg_agent)
    return wrapped_agent

def main(config,make_env):
    agent = make_agent(config)
    print("Running experiment")
    print(f"Config: {config}  \n")
    run_experiment(config, make_env,agent,multiprocess=False)
    print("Experiment complete")


if __name__ == "__main__":
    import argparse


    config_path = input("Enter the path to the config file: ")

    config = parse_config(config_path)

    save_path = input("Enter the path to save the results: ")
    exp_name = input("Enter the experiment name: ")

    # config["save_path"] = save_path
    config["experiment_name"] = exp_name
    config["results_dir"] = save_path

    main(config,make_env)

        
    
    