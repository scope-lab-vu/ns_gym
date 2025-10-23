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

"""
Continuous change in the cliffwalking environment 
"""


def read_config_file(config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config["gym_env"],config["wrapper"],config["agent"],config["experiment"]


########### Parse the config file and set up the logger ################################
config_file_path = "/media/--/home/n--/ns_gym/experiments/NS_CliffWalking/configs/PAMCTS_continous_config.yaml"

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


    env = gym.make("CliffWalking-v0",max_episode_steps=gym_config["max_episode_steps"])

    change_notification = wrapper_config["change_notification"]
    delta_change_notification = wrapper_config["delta_change_notification"]
    in_sim_change = wrapper_config["in_sim_change"]
    param_name = wrapper_config["param_name"]
    modified_rewards = wrapper_config["modified_rewards"]


    scheduler = nsb.schedulers.ContinuousScheduler(start=0,end=10)
    updateFn = nsb.update_functions.DistributionDecrmentUpdate(scheduler,k=0.02)
    params = {param_name:updateFn}
    realworld_env = nsb.wrappers.NSCliffWalkingWrapper(env,
                                                         params,
                                                         change_notification=change_notification,
                                                         delta_change_notification=delta_change_notification,
                                                         in_sim_change=in_sim_change,
                                                         modified_rewards=modified_rewards)
    return realworld_env


def run_all_experiments(num_iter,c,gamma,alpha,sample_id,q,seed,agent_config,wrapper_config,gym_config):
    """Run all experiments with the given gridsearch over parameters.
    """
    try:
        logger.info(f"STARTING: Experiment with p, num_iter:{num_iter} num_samples: {sample_id},seed: {seed},c {c}, gamma {gamma},alpha {alpha}")

        ########## Setup the environment ##########
        realworld_env = make_env( gym_config=gym_config,wrapper_config=wrapper_config)

        state_size =agent_config["state_size"]
        action_size = agent_config["action_size"]
        num_hidden_units = agent_config["num_hidden_units"]
        num_layers = agent_config["num_layers"]
        model_path = agent_config["model_path"]
        max_episode_steps = gym_config["max_episode_steps"]

        ############# Run the experiment #############
        episode_reward = 0
        start_time = time.time()
        done = False
        truncated = False
        obs,_= realworld_env.reset(seed=seed)
        count = 0
        
        
        DDQN_model = nsb.benchmark_algorithms.DDQN.DQN(state_size=state_size,action_size=action_size,seed=seed,num_hidden_units=num_hidden_units,num_layers=num_layers)
        PAMCTS_agent = nsb.benchmark_algorithms.PAMCTS(alpha=alpha,
                                                               mcts_iter=num_iter,
                                                               mcts_search_depth=agent_config["mcts_search_depth"],
                                                               mcts_discount_factor=gamma,
                                                               mcts_exploration_constant=c,
                                                               state_space_size=state_size,
                                                               action_space_size=action_size,
                                                               DDQN_model=DDQN_model,
                                                               DDQN_model_path=model_path)

        while not done and not truncated and count < max_episode_steps:
            planning_env = realworld_env.get_planning_env()
            action , _ = PAMCTS_agent.act(obs,planning_env,normalize=True) 
            obs,reward,done, truncated, info = realworld_env.step(action)
            episode_reward += reward.reward
            count += 1

        total_time = time.time() - start_time

        logger.info(f"FINISHED: Experiment withnum_iter:{num_iter} num_samples: {sample_id},seed: {seed},c {c}, gamma {gamma}")
        result = [sample_id,episode_reward,experiment_name,num_iter,total_time,seed,c,gamma,alpha]
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

    c = agent_config["c"]
    gamma = agent_config["gamma"]
    alpha = agent_config["alpha"]
    sample_id= [x for x in range(exp_config["num_samples"])]
    num_iter = exp_config["num_iter"]
    c = agent_config["c"]
    gamma = agent_config["gamma"]

    num_experiments = len(sample_id) * len(num_iter) * len(c) * len(gamma) * len(alpha)

    seeds = random.sample(range(100000), num_experiments)

    results_dir = os.path.join(script_dir, "results")

    os.makedirs(results_dir,exist_ok=True)
    outfile = os.path.join(results_dir,experiment_name + ".csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id","reward","experiment_name","num_iter","total_time","seed","c","gamma","alpha"])

    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile))
    writer_process.start()

    parameter_combinations = itertools.product(num_iter,c,gamma,alpha,sample_id)
    input = [(*params,queue,seeds[i],agent_config,wrapper_config,gym_config) for i,params in enumerate(parameter_combinations)]
    
    with Pool(15) as p:
        p.starmap(run_all_experiments, input)

    print("number of experiments",num_experiments)
    queue.put("DONE")
    writer_process.join()


if __name__ == "__main__":
    print("Starting experiment")
    main()
    print("Experiment finished")



