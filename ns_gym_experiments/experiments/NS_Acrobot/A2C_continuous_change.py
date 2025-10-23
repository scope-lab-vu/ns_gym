import ns_gym.base
from ns_gym.eval import run_experiment
from ns_gym.utils import parse_config  
import gymnasium as gym
import numpy as np

import torch

from stable_baselines3 import A2C

def make_env(config):
    # Increase link length to 1.5
    env = gym.make(config["env_name"])

    tunable_param = config["tunable_params"]
    scheduler = ns_gym.schedulers.ContinuousScheduler()

    update_fn = ns_gym.update_functions.IncrementUpdate(scheduler,0.1)
    # update_fn = ns_gym.update_functions.DecrementUpdate(scheduler,0.5)
    param_map  = {tunable_param:update_fn}
    env = ns_gym.wrappers.NSClassicControlWrapper(env,param_map)
    return env

def make_agent(config):
    model_path = config["model_path"]
    agent = A2C.load(model_path)
    wrapped_agent = ns_gym.base.StableBaselineWrapper(agent)
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

        
    
    