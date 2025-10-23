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
import multiprocessing
import datetime
import itertools


"""
Reproducing the "Vanilla MCTS" experiments in ADAMCTS paper. 

MCTS does not konw the ground truth transition function. It only knows MDP_{k-1}
"""
current_datetime = datetime.datetime.now()
formatted_date = current_datetime.strftime('%Y-%m-%d_%H:%M:%S')
os.makedirs("logs",exist_ok=True)
logsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs")
log_name = f"VanillaMCTS_FrozenLake_ChangeNotif_True_{formatted_date}.log"

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                handlers=[logging.FileHandler(os.path.join(logsdir,log_name), mode='w')])
logger = logging.getLogger()

def make_env(p,change_notification,delta_change_notification,in_sim_change):
    """Make non-stationary FrozenLake environment.
    """
    env = gym.make('FrozenLake-v1',is_slippery = False,render_mode = "ansi",max_episode_steps=1000)
    fl_scheduler = nsb.schedulers.DiscreteScheduler({0})
    fl_updateFn = nsb.update_functions.DistributionStepWiseUpdate(fl_scheduler,[[p,(1-p)/2,(1-p)/2]])
    param = {"P":fl_updateFn}

    realworld_env = nsb.wrappers.NSFrozenLakeWrapper(env,
                                                     param,
                                                     change_notification = change_notification, 
                                                     delta_change_notification = delta_change_notification, 
                                                     in_sim_change = in_sim_change,
                                                     initial_prob_dist=[0.7,0.15,0.15])
    return realworld_env

def run_single_experiment(p, gamma, c, num_iter, sample_id):
    """Run a single experiment with the given parameters.
    Args:
        p (float): Probability of the agent moving in the intended direction.
        gamma (float): Discount factor.
        c (float): Exploration constant.
        num_iter (int): Number of iterations.
        num_samples (int): Number of samples/seeds to run on this trail.

    Returns:
        dict: Dictionary containing the results of the experiment.
    """
    realworld_env = make_env(p = p, change_notification = False, delta_change_notification = False, in_sim_change = False)
    mcts_agent = nsb.benchmark_algorithms.StochasticMCTS(gamma=gamma,num_iter=num_iter,c = c)    
    obs,_ = realworld_env.reset(seed=np.random(sample_id))
    done = False
    planning_env = realworld_env.get_planning_env()
    episode_reward = 0
    while not done:
        action = mcts_agent.get_action(observation=obs.state,env=planning_env)
        obs,reward,done, _, info = realworld_env.step(action)
        mcts_agent = nsb.benchmark_algorithms.StochasticMCTS(gamma=gamma,num_iter=num_iter,c = c)   
        planning_env = realworld_env.get_planning_env()
        episode_reward += reward.reward
    return episode_reward

def run_all_experiments(p, gamma, c, num_iter, sample_id,q):
    """Run all experiments with the given gridsearch over parameters.
    Args:
        p (list): List of probabilities.
        gamma (list): List of discount factors.
        c (list): List of exploration constants.
        num_iter (list): List of number of iterations.
        num_samples (list): List of number of samples.
        q (multiprocessing.Queue): Queue to store the results.
    """
    logger.setLevel(logging.INFO)
    try:
        logger.info(f"Vanilla MCTS no change notif. experiment with p: {p}, gamma: {gamma}, c: {c}, num_iter: {num_iter}, num_samples: {sample_id}")
        start_time = time.time()
        episode_reward = run_single_experiment(p, gamma, c, num_iter, sample_id)
        total_time = time.time() - start_time
        result = [sample_id,episode_reward,"Vanilla MCTS no change notif.",p,gamma,c,num_iter]
        q.put(result)

    except Exception as e:
        logger.error(f"Error in running experiment: {e}")
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
        print(f"Error in writing results to file: {e}")

def main():
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    # p = [0.4, 0.5, 0.6, 0.8, 0.9, 1]
    p = [0.4]
    gamma = [0.999]
    c = [np.sqrt(2)]
    num_iter = [1000]
    sample_id= [x for x in range(100)]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir,exist_ok=True)
    current_datetime = datetime.datetime.now()
    formatted_date = current_datetime.strftime('%Y-%m-%d_%H:%M:%S')
    outfile = os.path.join(results_dir,f"VanillaMCTS_no_change_notif_{formatted_date}.csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id","reward","experiment_name","p","gamma","c","num_iter","total_time","seed"])

    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile))
    writer_process.start()

    parameter_combinations = itertools.product(p, gamma, c, num_iter, sample_id)
    
    with Pool(16) as p:
        p.starmap(run_all_experiments,[(*params,queue) for params in parameter_combinations])
    
    queue.put("DONE")
    writer_process.join()


if __name__ == "__main__":
    print("Starting Exp")
    main()
  














