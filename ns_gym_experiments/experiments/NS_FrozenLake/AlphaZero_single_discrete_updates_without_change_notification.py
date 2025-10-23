
import os
import csv
import numpy as np
import gymnasium as gym
import ns_gym as nsb 
from ns_gym.benchmark_algorithms.AlphaZero.alphazero import AlphaZeroAgent,AlphaZeroNetwork
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
Reproducing the Alphazero experiments where frozen lake environment is non-stationary and there is no change notification. The agent is trained on a stationary environment and tested on a non-stationary environment. 
The agent is not aware of the change in the environment. Splipery starts at 0.7 and changes among  [0.4, 0.5, 0.6, 0.8, 0.9, 1]. 

MCTS does not konw the ground truth transition function. It only knows MDP_{k-1}
"""
def read_config_file(config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config["gym_env"],config["wrapper"],config["alphazero_agent"],config["experiment"]

def make_env(p,gym_config,wrapper_config):
    """Make non-stationary FrozenLake environment.
    """
    env = gym.make(gym_config["name"],is_slippery = gym_config["is_slippery"],render_mode = gym_config["render_mode"],max_episode_steps=gym_config["max_episode_steps"])
    fl_scheduler = nsb.schedulers.DiscreteScheduler({0})
    fl_updateFn = nsb.update_functions.DistributionStepWiseUpdate(fl_scheduler,[[p,(1-p)/2,(1-p)/2]])
    param = {"P":fl_updateFn}

    realworld_env = nsb.wrappers.NSFrozenLakeWrapper(env,
                                                     param,
                                                     change_notification = wrapper_config["change_notification"], 
                                                     delta_change_notification = wrapper_config["delta_change_notification"], 
                                                     in_sim_change = wrapper_config["in_sim_change"],
                                                     initial_prob_dist=wrapper_config["initial_prob_dist"],
                                                     modified_rewards = wrapper_config["modified_rewards"])
    return realworld_env

# def run_single_experiment(p,sample_id,seed):
#     """Run a single experiment with the given parameters.
#     Args:
#         p (float): Probability of the agent moving in the intended direction.
#         gamma (float): Discount factor.
#         c (float): Exploration constant.
#         num_iter (int): Number of iterations.
#         num_samples (int): Number of samples/seeds to run on this trail.

#     Returns:
#         dict: Dictionary containing the results of the experiment.
#     """
#     realworld_env = make_env(p = p, change_notification = False, delta_change_notification = False, in_sim_change = False)
#     obs,_ = realworld_env.reset(seed=seed)
#     done = False
#     alphazero_agent = AlphaZeroAgent(
#                                     action_space_dim=4,
#                                     observation_space_dim=16,
#                                     model = AlphaZeroNetwork
#                                     lr=0.001,
#                                     mcts=mcts_agent,
#                                     n_hidden_layers=2,
#                                     n_hidden_units=64,
#                                     gamma=0.99,
#                                     c=np.sqrt(2),
#                                     num_mcts_simulations=500,
#                                     max_mcts_search_depth=500,
#                                     model_checkpoint_path = None
#                                 )
#     episode_reward = 0
#     truncated = False
#     while not done and not truncated:
#         action = DDQN_agent.act(obs)
#         obs,reward,done, truncated, info = realworld_env.step(action)
#         episode_reward += reward.reward
#     return episode_reward

def run_all_experiments(p,num_iter,c,gamma,sample_id,q,seed,agent_config,wrapper_config,gym_config):
    """Run all experiments with the given gridsearch over parameters.
    Args:
        p (list): List of probabilities.
        num_samples (list): List of number of samples.
        q (multiprocessing.Queue): Queue to store the results.
    """
    try:
        logger.info(f"STARTING: Experiment with p: {p}, num_iter:{num_iter} num_samples: {sample_id},seed: {seed}")

        ########## Setup the environment ##########
        realworld_env = make_env(p = p, gym_config=gym_config,wrapper_config=wrapper_config)

        alphazero_agent = AlphaZeroAgent(action_space_dim=agent_config["action_space_dim"],
                                         observation_space_dim=agent_config["observation_space_dim"],
                                         model=AlphaZeroNetwork,
                                         lr=agent_config["lr"],
                                         n_hidden_layers=agent_config["n_hidden_layers"],
                                         n_hidden_units=agent_config["n_hidden_units"],
                                         gamma=gamma,
                                         c=c,
                                         num_mcts_simulations=num_iter,
                                         max_mcts_search_depth=agent_config["max_mcts_search_depth"],
                                         model_checkpoint_path = agent_config["model_checkpoint_path"])

        ############# Run the experiment #############
        episode_reward = 0
        start_time = time.time()
        done = False
        truncated = False
        obs,_= realworld_env.reset(seed=seed)
        while not done and not truncated:
            planning_env = realworld_env.get_planning_env()
            action = alphazero_agent.act(obs,planning_env)
            obs,reward,done, truncated, info = realworld_env.step(action)
            episode_reward += reward.reward

        total_time = time.time() - start_time

        logger.info(f"FINISHED: Experiment with p: {p}, num_samples: {sample_id},seed: {seed} in {total_time} seconds.")
        result = [sample_id,episode_reward,experiment_name,p,num_iter,total_time,seed]
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

    p = exp_config["p"]
    sample_id= [x for x in range(exp_config["num_samples"])]
    num_iter = exp_config["num_iter"]
    c = agent_config["c"]
    gamma = agent_config["gamma"]

    num_experiments = exp_config["num_experiments"]

    seeds = random.sample(range(100000), num_experiments)

    results_dir = os.path.join(script_dir, "results")

    os.makedirs(results_dir,exist_ok=True)
    outfile = os.path.join(results_dir,experiment_name + ".csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id","reward","experiment_name","p","num_iter","total_time","seed"])

    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile))
    writer_process.start()

    parameter_combinations = itertools.product(p,num_iter,c,gamma,sample_id)
    input = [(*params,queue,seeds[i],agent_config,wrapper_config,gym_config) for i,params in enumerate(parameter_combinations)]
    
    with Pool(multiprocessing.cpu_count()) as p:
        p.starmap(run_all_experiments, input)

    print("number of experiments",num_experiments)
    queue.put("DONE")
    writer_process.join()


if __name__ == "__main__":
    print("Starting Exp")
    config_file_path = "/home/cc/ns_gym/experiments/NS_FrozenLake/alphazero_eval_config.yaml"

    gym_config,wrapper_config,agent_config,exp_config = read_config_file(config_file_path)

    current_datetime = datetime.datetime.now()
    formatted_date = current_datetime.strftime('%Y-%m-%d_%H:%M:%S')
    experiment_name = f"AlphaZero_singe_discrete_change_withoutChangNotif_{formatted_date}"
    os.makedirs("logs",exist_ok=True)
    logsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs")
    log_name = experiment_name + ".log"

    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(logsdir,log_name), mode='w')])
    logger = logging.getLogger()

    script_path = Path(__file__)
    script_dir = script_path.parent
    main()













