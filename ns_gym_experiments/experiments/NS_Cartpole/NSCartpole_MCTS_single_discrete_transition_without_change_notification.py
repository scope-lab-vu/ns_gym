import json 
import os
import sys
import argparse
import csv
import numpy as np
import gymnasium as gym
import ns_gym as nsb 
import time
import logging
import pickle
from multiprocessing import Pool
import random
import multiprocessing
import datetime
import itertools


"""
Reproducing the "Vanilla MCTS" experiments on cartpole env in PAMCTS paper. The NSMDP has one transition from MDP_0 to MDP_1 at MDP time step 0. 

In this experiment we vary pole mass amoung the following values: 0.1, 1.0, 1.2 , 1.3, 1.5 , The initial pole mass is 0.1. 
WE vary MCTS iterations among { 25, 50, 75, 100, 200, 300 }.
MCTS does not know the ground truth transition function. It only knows MDP_{k-1}
"""

current_datetime = datetime.datetime.now()
formatted_date = current_datetime.strftime('%Y-%m-%d_%H:%M:%S')
experiment_name = f"CartpoleVanillaMCTS_withoutChangNotif_{formatted_date}"

logsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs")
os.makedirs(logsdir,exist_ok=True)
log_name = experiment_name + ".log"

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                handlers=[logging.FileHandler(os.path.join(logsdir,log_name), mode='w')])
logger = logging.getLogger()

def make_env(mass,change_notification,delta_change_notification,in_sim_change):
    """Make non-stationary FrozenLake environment.
    """
    env = gym.make("CartPole-v1",max_episode_steps=2500)
    mass_pole_scheduler = nsb.schedulers.DiscreteScheduler({0})
    mass_pole_updateFn = nsb.update_functions.StepWiseUpdate(mass_pole_scheduler,[mass])
    params = {"masspole":mass_pole_updateFn}
    realworld_env = nsb.wrappers.NSClassicControlWrapper(env,
                                                         params,
                                                         change_notification=change_notification,
                                                         delta_change_notification=delta_change_notification,
                                                         in_sim_change=in_sim_change)
    return realworld_env

def run_single_experiment(mass, gamma, c, num_iter, sample_id,seed):
    """Run a single experiment with the given parameters.
    Args:
        m (float): Mass of the pole
        gamma (float): Discount factor.
        c (float): Exploration constant.
        num_iter (int): Number of iterations.
        num_samples (int): Number of samples/seeds to run on this trail.

    Returns:
        dict: Dictionary containing the results of the experiment.
    """
    realworld_env = make_env(mass=mass, change_notification = False, delta_change_notification = False, in_sim_change = False)
    obs,_ = realworld_env.reset(seed = seed)
    done = False
    episode_reward = 0
    while not done:
        planning_env = realworld_env.get_planning_env()
        mcts_agent = nsb.benchmark_algorithms.MCTS(env=planning_env, state=obs.state, gamma=gamma, d=500, m=num_iter, c=c)
        action,_ = mcts_agent.search()
        obs,reward,done, _, info = realworld_env.step(action)
        episode_reward += reward.reward
    return episode_reward

def run_all_experiments(mass, gamma, c, num_iter, sample_id,q,seed):
    """Run all experiments with the given gridsearch over parameters.
    Args:
        mass (list): list of masses of probabilities.
        gamma (list): List of discount factors.
        c (list): List of exploration constants.
        num_iter (list): List of number of iterations.
        num_samples (list): List of number of samples.
        q (multiprocessing.Queue): Queue to store the results.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # logs_dir = os.path.join(script_dir, "logs")
    # os.makedirs(logs_dir,exist_ok=True)
    # pid = os.getpid()
    # logger = logging.getLogger(f'worker_{pid}')
    # handler = logging.FileHandler(os.path.join(logs_dir,f"log_worker_{pid}.log"))
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    # logger.setLevel(logging.INFO)
    try:
        logger.info(f"STARTED: Carpole Vanilla MCTS no change notit. experiment with pole mass: {mass}, gamma: {gamma}, c: {c}, num_iter: {num_iter}, num_samples: {sample_id}")
        start_time = time.time()
        episode_reward = run_single_experiment(mass, gamma, c, num_iter, sample_id,seed)
        total_time = time.time() - start_time
        result = [sample_id,episode_reward,"Vanilla MCTS no change notif,negative reward.",mass,gamma,c,num_iter,total_time,seed]
        logger.info(f"FINISHED: Carpole Vanilla MCTS no change notit. experiment with pole mass: {mass}, gamma: {gamma}, c: {c}, num_iter: {num_iter}, num_samples: {sample_id}, in {total_time} seconds.")
        q.put(result)

    except Exception as e:
        logger.error(f"Error in running experiment: {e}", exc_info=True)
        print(f"Error in running experiment: {e}")
        return None
    
def write_results_to_file(q,outfile):
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
    except Exception as e:
        logger.error(f"Error in writing results to file: {e}", exc_info=True)
        print(f"Error in writing results to file: {e}")

def main():
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    mass = [0.1, 1.0, 1.2, 1.3, 1.5]
    gamma = [0.5]
    c = [np.sqrt(2)]
    num_iter = [50,100,300]
    sample_id= [x for x in range(50)]
    num_experiments = len(mass) * len(gamma) * len(c) * len(num_iter) * len(sample_id)
    seeds = random.sample(range(1000000), num_experiments)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir,exist_ok=True)
    outfile = os.path.join(results_dir,experiment_name + ".csv")


    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id","reward","experiment_name","polemass","gamma","c","num_iter","total_episode_time","seed"])


    parameter_combinations = itertools.product(mass, gamma, c, num_iter, sample_id)
    input  = [(*params,queue,seeds[i]) for i,params in enumerate(parameter_combinations)]


    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile))
    writer_process.start()

    with Pool(multiprocessing.cpu_count()-1) as p:
        p.starmap(run_all_experiments,input)
    
    queue.put("DONE")
    writer_process.join()


if __name__ == "__main__":
    print("Starting Exp")
    main()
  














