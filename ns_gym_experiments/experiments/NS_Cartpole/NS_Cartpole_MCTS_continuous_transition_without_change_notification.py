import os
import csv
import numpy as np
import gymnasium as gym
import ns_gym as nsb 
from ns_gym.benchmark_algorithms import MCTS
import time
import logging
from pathlib import Path
import random
from multiprocessing import Pool
import multiprocessing
import datetime
import itertools
import yaml




def read_config_file(config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config["gym_env"],config["wrapper"],config["agent"],config["experiment"]


########### Parse the config file and set up the logger ################################
config_file_path = "----"

gym_config,wrapper_config,agent_config,exp_config = read_config_file(config_file_path)


script_path = Path(__file__)
script_dir = script_path.parent

current_datetime = datetime.datetime.now()
formatted_date = current_datetime.strftime('%Y-%m-%d')
experiment_name = exp_config["experiment_name"]+f"_{formatted_date}"

logsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs")
log_name = experiment_name + ".log"
os.makedirs(logsdir,exist_ok=True)
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                handlers=[logging.FileHandler(os.path.join(logsdir,log_name), mode='w')])
logger = logging.getLogger()

##########################################################################################


def make_env(gym_config,wrapper_config):
    """Make non-stationary Classic Control environment.
    """


    env = gym.make("CartPole-v1",max_episode_steps=gym_config["max_episode_steps"])

    change_notification = wrapper_config["change_notification"]
    delta_change_notification = wrapper_config["delta_change_notification"]
    in_sim_change = wrapper_config["in_sim_change"]
    param_name = wrapper_config["param_name"] 

    scheduler = nsb.schedulers.ContinuousScheduler()
    updateFn = nsb.update_functions.IncrementUpdate(scheduler=scheduler,k=0.1)
    params = {param_name:updateFn}
    realworld_env = nsb.wrappers.NSClassicControlWrapper(env,
                                                         params,
                                                         change_notification=change_notification,
                                                         delta_change_notification=delta_change_notification,
                                                         in_sim_change=in_sim_change)
    return realworld_env


def run_all_experiments(num_iter,c,gamma,sample_id,q,seed,agent_config,wrapper_config,gym_config):
    """Run all experiments with the given gridsearch over parameters.
    """
    try:
        logger.info(f"STARTING: Experiment with p, num_iter:{num_iter} num_samples: {sample_id},seed: {seed},c {c}, gamma {gamma}")

        ########## Setup the environment ##########
        realworld_env = make_env( gym_config=gym_config,wrapper_config=wrapper_config)

        ############# Run the experiment #############
        episode_reward = 0
        start_time = time.time()
        done = False
        truncated = False
        obs,_= realworld_env.reset(seed=seed)

        while not done and not truncated:
            planning_env = realworld_env.get_planning_env()
            mcts_agent = nsb.benchmark_algorithms.MCTS(planning_env,obs,agent_config["max_mcts_search_depth"],m=num_iter,c=c,gamma=gamma)
            action,_ = mcts_agent.search()
            obs,reward,done, truncated, info = realworld_env.step(action)
            episode_reward += reward

        total_time = time.time() - start_time

        logger.info(f"FINISHED: Experiment withnum_iter:{num_iter} num_samples: {sample_id},seed: {seed},c {c}, gamma {gamma}")
        result = [sample_id,episode_reward,experiment_name,num_iter,total_time,seed,c,gamma]
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





    sample_id= [x for x in range(exp_config["num_samples"])]
    num_iter = exp_config["num_iter"]
    c = agent_config["c"]
    gamma = agent_config["gamma"]

    num_experiments = len(sample_id) * len(num_iter) * len(c) * len(gamma)

    seeds = random.sample(range(100000), num_experiments)

    results_dir = os.path.join(script_dir, "results")

    os.makedirs(results_dir,exist_ok=True)
    outfile = os.path.join(results_dir,experiment_name + ".csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id","reward","experiment_name","num_iter","total_time","seed","c","gamma"])

    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile))
    writer_process.start()

    parameter_combinations = itertools.product(num_iter,c,gamma,sample_id)
    input = [(*params,queue,seeds[i],agent_config,wrapper_config,gym_config) for i,params in enumerate(parameter_combinations)]
    
    with Pool(multiprocessing.cpu_count()) as p:
        p.starmap(run_all_experiments, input)

    print("number of experiments",num_experiments)
    queue.put("DONE")
    writer_process.join()


if __name__ == "__main__":
    print("Starting experiment")

    main()
    print("Experiment finished")



