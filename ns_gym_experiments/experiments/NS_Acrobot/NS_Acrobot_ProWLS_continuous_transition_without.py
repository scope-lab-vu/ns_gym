import sys
sys.path.append('/data/home/ifb5104/ns_gym')

import os
import csv
import numpy as np
import gymnasium as gym
import ns_gym as nsb 
from pathlib import Path
import random
import itertools
import yaml
import torch
from multiprocessing import Pool, Manager, Process
import datetime
import time
import logging
from torch import optim


from ns_gym.benchmark_algorithms.ProWLS_ns import ProWLSAgent, Config

def get_optimizer(optim_name):
    if optim_name == "Adam":
        return optim.Adam
    elif optim_name == "SGD":
        return optim.SGD
    else:
        return optim.Adam

def read_config_file(config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config["gym_env"], config["wrapper"], config["agent"], config["experiment"]

########### Parse the config file and set up the logger ################################
config_file_path = 'experiments/NS_Acrobot/Acrobot_ProWLS_continuous_update.yaml'

gym_config, wrapper_config, agent_config, exp_config = read_config_file(config_file_path)

script_path = Path(__file__)
script_dir = script_path.parent


current_datetime = datetime.datetime.now()
formatted_date = current_datetime.strftime('%Y-%m-%d')
experiment_name = exp_config["experiment_name"] + f"_{formatted_date}"

logsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
log_name = experiment_name + ".log"
os.makedirs(logsdir, exist_ok=True)
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(logsdir, log_name), mode='w')])
logger = logging.getLogger()

##########################################################################################

def make_env(gym_config, wrapper_config):
    """Make non-stationary Classic Control environment."""

    env = gym.make("Acrobot-v1", max_episode_steps=gym_config["max_episode_steps"])

    change_notification = wrapper_config["change_notification"]
    delta_change_notification = wrapper_config["delta_change_notification"]
    in_sim_change = wrapper_config["in_sim_change"]
    param_name = wrapper_config["param_name"]

    scheduler = nsb.schedulers.ContinuousScheduler()
    updateFn = nsb.update_functions.IncrementUpdate(scheduler=scheduler, k=0.1)
    params = {param_name: updateFn}
    realworld_env = nsb.wrappers.NSClassicControlWrapper(env,
                                                         params,
                                                         change_notification=change_notification,
                                                         delta_change_notification=delta_change_notification,
                                                         in_sim_change=in_sim_change)

    return realworld_env

def run_all_experiments(num_iter, c, gamma, alpha, sample_id, q, seed, agent_config, wrapper_config, gym_config):
    """Run all experiments with the given gridsearch over parameters."""
    try:
        print(f"STARTING: Experiment with p, num_iter:{num_iter} num_samples: {sample_id},seed: {seed},c {c}, gamma {gamma},alpha {alpha}")
        logger.info(f"STARTING: Experiment with p, num_iter:{num_iter} num_samples: {sample_id},seed: {seed},c {c}, gamma {gamma},alpha {alpha}")

        ########## Setup the environment ##########
        
        realworld_env = make_env(gym_config=gym_config, wrapper_config=wrapper_config)
        model_path = agent_config["model_path"]

        ############# Run the experiment #############
        
        start_time = time.time()
        done = False
        truncated = True 
        
        
        # Wrap agent_config in a Config object
        agent_config['env'] = realworld_env
        agent_config['device'] = torch.device(agent_config['device'])
        agent_config['optim'] = get_optimizer(agent_config['optim'])
        agent_config['max_horizon'] = gym_config["max_episode_steps"] 
        config = Config(agent_config)

        # Load the trained ProWLS model
        pro_wls_agent = ProWLSAgent(config)
        pro_wls_agent.actor.load_state_dict(torch.load(model_path))
        pro_wls_agent.actor.eval()
        episode_reward = 0
        state, _ =  realworld_env.reset(seed=random.randint(0, 100000))

        t = 0
        while True:
            action, _, _ = pro_wls_agent.get_action(state)
        
            next_state, reward, done, truncated, _ = realworld_env.step(action)
            if isinstance(reward, nsb.base.Reward):
                reward = reward.reward
            state = next_state
            episode_reward += reward
            if done or truncated or t > 500:
                break
            t += 1
        

        total_time = time.time() - start_time

        logger.info(f"FINISHED: Experiment with num_iter:{num_iter} num_samples: {sample_id},seed: {seed},c {c}, gamma {gamma}")
        
        result = [sample_id, episode_reward, experiment_name, num_iter, total_time, seed, c, gamma, alpha]
        q.put(result)
    
    except Exception as e:
        logger.error(f"Error in running experiment: {e}", exc_info=True)
        print(f"Error in running experiment: {e}")
        return None


def write_results_to_file(q, outfile):
    """Write results to file.
    Args:
        q (multiprocessing.Queue): Queue containing the results.
        outfile (str): Path to the output file.
    """
    try:
        with open(outfile, mode='a', newline='') as file:
            writer = csv.writer(file)
            while True:
                result = q.get()
                if result == 'DONE':  # Check for sentinel value
                    break
                writer.writerow(result)
        logger.info(f"Results written to file")
    except Exception as e:
        print(f"Error in writing results to file: {e}")

def main():
    manager = Manager()
    queue = manager.Queue()

    c = agent_config["c"]
    gamma = agent_config["gamma"]
    alpha = agent_config["alpha"]

    sample_id = [x for x in range(exp_config["num_samples"])]
    num_iter = exp_config["num_iter"]
    c = agent_config["c"]
    gamma = agent_config["gamma"]

    num_experiments = len(sample_id) * len(num_iter) * len(c) * len(gamma) * len(alpha)
    seeds = random.sample(range(100000), num_experiments)

    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    outfile = os.path.join(results_dir, experiment_name + ".csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "reward", "experiment_name", "num_iter", "total_time", "seed", "c", "gamma", "alpha"])

    writer_process = Process(target=write_results_to_file, args=(queue, outfile))
    writer_process.start()

    parameter_combinations = itertools.product(num_iter, c, gamma, alpha, sample_id)
    input = [(*params, queue, seeds[i], agent_config, wrapper_config, gym_config) for i, params in enumerate(parameter_combinations)]
    print("number of experiments", num_experiments)

    # with Pool(10) as p:
    #     p.starmap(run_all_experiments, input)
    
    run_all_experiments(*input[0])

    

    queue.put("DONE")
    writer_process.join()

if __name__ == "__main__":
    print("Starting experiment")
    main()
    print("Experiment finished")
