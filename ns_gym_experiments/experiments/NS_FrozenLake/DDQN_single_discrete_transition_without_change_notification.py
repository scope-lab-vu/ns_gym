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
Reproducing the DDQN experiments where frozen lake environment is non-stationary and there is no change notification. The agent is trained on a stationary environment and tested on a non-stationary environment. 
The agent is not aware of the change in the environment. Splipery starts at 0.7 and changes among  [0.4, 0.5, 0.6, 0.8, 0.9, 1]. 

MCTS does not konw the ground truth transition function. It only knows MDP_{k-1}
"""

current_datetime = datetime.datetime.now()
formatted_date = current_datetime.strftime('%Y-%m-%d_%H:%M:%S')
experiment_name = f"DDQN_singe_discrete_change_withoutChangNotif_{formatted_date}"
os.makedirs("logs",exist_ok=True)
logsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs")
log_name = experiment_name + ".log"

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                handlers=[logging.FileHandler(os.path.join(logsdir,log_name), mode='w')])
logger = logging.getLogger()

script_path = Path(__file__)
script_dir = script_path.parent
model_path = script_dir  / ".." / ".." / "ns_gym" / "benchmark_algorithms" / "DDQN" / "DDQN_models" / "FrozenLake_DDQN_07.pth"

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

def run_single_experiment(p,sample_id,seed):
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
    obs,_ = realworld_env.reset(seed=seed)
    done = False
    DDQN_model = nsb.benchmark_algorithms.DDQN.DQN(state_size=16,action_size=4,seed=seed,num_hidden_units=64,num_layers=3)
    DDQN_agent = nsb.benchmark_algorithms.DDQN.DQNAgent(state_size=16,action_size=4,seed=seed,model=DDQN_model,model_path=model_path)
    episode_reward = 0
    truncated = False
    while not done and not truncated:
        action = DDQN_agent.act(obs)
        obs,reward,done, truncated, info = realworld_env.step(action)
        episode_reward += reward.reward
    return episode_reward

def run_all_experiments(p, sample_id,q,seed):
    """Run all experiments with the given gridsearch over parameters.
    Args:
        p (list): List of probabilities.
        num_samples (list): List of number of samples.
        q (multiprocessing.Queue): Queue to store the results.
    """
    try:
        logger.info(f"STARTING: Experiment with p: {p}, num_samples: {sample_id},seed: {seed}")
        start_time = time.time()
        episode_reward = run_single_experiment(p, sample_id,seed=seed)
        total_time = time.time() - start_time
        logger.info(f"FINISHED: Experiment with p: {p}, num_samples: {sample_id},seed: {seed} in {total_time} seconds.")
        result = [sample_id,episode_reward,experiment_name,p,total_time,seed]
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
    p = [0.4, 0.5, 0.6, 0.8, 0.9, 1]
    sample_id= [x for x in range(100)]
    num_experiments = len(p) * len(sample_id)
    seeds = random.sample(range(10000), num_experiments)

    results_dir = os.path.join(script_dir, "results")

    os.makedirs(results_dir,exist_ok=True)
    outfile = os.path.join(results_dir,experiment_name + ".csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id","reward","experiment_name","p","total_time","seed"])

    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile))
    writer_process.start()

    parameter_combinations = itertools.product(p,sample_id)
    input = [(*params,queue,seeds[i]) for i,params in enumerate(parameter_combinations)]
    
    with Pool(multiprocessing.cpu_count()-2) as p:
        p.starmap(run_all_experiments, input)

    print("number of experiments",num_experiments)
    queue.put("DONE")
    writer_process.join()


if __name__ == "__main__":
    print("Starting Exp")
    main()













