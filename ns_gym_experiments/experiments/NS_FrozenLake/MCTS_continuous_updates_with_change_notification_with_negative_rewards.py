import os
import csv
import numpy as np
import gymnasium as gym
import ns_gym as nsb 
import time
import logging
from pathlib import Path
import random
from multiprocessing import Pool
import multiprocessing
import datetime
import itertools


"""
Vanilla MCTS where the agent is trained on a stationary environment and tested on a non-stationary environment. 
The agent is not aware of the change in the environment. The environment begins with probability of moving in the intended direction 

MCTS does not konw the ground truth transition function. It only knows MDP_{0}
"""


current_datetime = datetime.datetime.now()
formatted_date = current_datetime.strftime('%Y-%m-%d_%H:%M:%S')
experiment_name = f"MCTS_ContinuousUpdate_with_ChangNotif_{formatted_date}"
os.makedirs("logs",exist_ok=True)
logsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs")
log_name = experiment_name + ".log"

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                handlers=[logging.FileHandler(os.path.join(logsdir,log_name), mode='w')])
logger = logging.getLogger()

script_path = Path(__file__)
script_dir = script_path.parent


def make_env(change_notification,delta_change_notification,in_sim_change):
    """Make non-stationary FrozenLake environment.
    """
    env = gym.make('FrozenLake-v1',is_slippery = False,render_mode = "ansi",max_episode_steps=1000)
    fl_scheduler = nsb.schedulers.ContinuousScheduler(start=1,end = 3)
    fl_updateFn = nsb.update_functions.DistributionDecrmentUpdate(fl_scheduler,k=0.2)
    param = {"P":fl_updateFn}

    realworld_env = nsb.wrappers.NSFrozenLakeWrapper(env,
                                                     param,
                                                     change_notification = change_notification, 
                                                     delta_change_notification = delta_change_notification, 
                                                     in_sim_change = in_sim_change,
                                                     initial_prob_dist=[1,0,0],
                                                     modified_rewards={"H":-1,"G":1,"F":0,"S":0})
    return realworld_env


def run_single_experiment(gamma, c, num_iter, sample_id,seed):
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
    realworld_env = make_env(change_notification = True, delta_change_notification = False, in_sim_change = False)
    obs,_ = realworld_env.reset(seed=seed)
    done = False
    episode_reward = 0
    truncated = False
    while not done and not truncated:
        planning_env = realworld_env.get_planning_env()
        assert(planning_env.unwrapped.P == realworld_env.unwrapped.P)
        mcts_agent = nsb.benchmark_algorithms.MCTS(env=planning_env,state=obs.state,d=500, m=num_iter, c=c, gamma=gamma)
        action,action_vals = mcts_agent.search()
        obs,reward,done, truncated, info = realworld_env.step(action)
        episode_reward += reward.reward
    return episode_reward

def run_all_experiments(gamma, c, num_iter, sample_id,q,seed):
    """Run all experiments with the given gridsearch over parameters.
    Args:
        p (list): List of probabilities.
        gamma (list): List of discount factors.
        c (list): List of exploration constants.
        num_iter (list): List of number of iterations.
        num_samples (list): List of number of samples.
        q (multiprocessing.Queue): Queue to store the results.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        logger.info(f"STARTING: Experiment with gamma: {gamma}, c: {c}, num_iter: {num_iter}, num_samples: {sample_id},seed: {seed}")
        start_time = time.time()
        episode_reward = run_single_experiment(gamma, c, num_iter, sample_id,seed=seed)
        total_time = time.time() - start_time
        logger.info(f"FINISHED: Experiment with gamma: {gamma}, c: {c}, num_iter: {num_iter}, num_samples: {sample_id},seed: {seed} in {total_time} seconds.")
        result = [sample_id,episode_reward,experiment_name,gamma,c,num_iter,total_time,seed]
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
        logger.info(f"Results written to file")
    except Exception as e:
        print(f"Error in writing results to file: {e}")

def main():
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    gamma = [0.99]
    c = [np.sqrt(2)]
    num_iter = [25, 100, 1000, 1000]
    sample_id= [x for x in range(100)]
    num_experiments = len(gamma) * len(c) * len(num_iter) * len(sample_id)
    seeds = random.sample(range(100000), num_experiments)
    #seeds = np.random.randint(0, 10000, size=num_experiments)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir,exist_ok=True)
    outfile = os.path.join(results_dir,experiment_name + ".csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id","reward","experiment_name","gamma","c","num_iter","total_time","seed"])

    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile))
    writer_process.start()

    parameter_combinations = itertools.product(gamma, c, num_iter, sample_id)
    input = [(*params,queue,seeds[i]) for i,params in enumerate(parameter_combinations)]
    
    with Pool(multiprocessing.cpu_count()-2) as p:
        p.starmap(run_all_experiments, input)

    print("number of experiments",num_experiments)
    queue.put("DONE")
    writer_process.join()


if __name__ == "__main__":
    print("Starting Exp")
    main()
    

