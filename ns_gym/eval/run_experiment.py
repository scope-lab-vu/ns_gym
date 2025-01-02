import ns_gym
import gymnasium as gym
from ns_gym.utils import parse_config
import random
import os
import csv
import multiprocessing
from multiprocessing import Pool
import logging
import itertools
import time
"""
Generic experiment runner functions
- Reads a YAML config file or command line arguments
- Builds non-stationary env
- Configures benchmark algorithms
- Stores results csv, trace visualizer
"""


def make_env(config):
    pass

def run_episode(queue,env,agent,seed,sample_id,config,logger):
    """Run an episode with a given agent and environment.
    """

    try:
        logger.info(f"Running experiment with seed {seed}")
        done = False
        truncated = False
        obs,_ = env.reset(seed)
        
        episode_reward = []
        
        num_steps = 0
        start_time = time.time()    

        while not done and not truncated:
            action = agent.act(obs)
            obs, reward, done, truncated = env.step(action)
            obs,reward = ns_gym.utils.type_mismatch_checker(obs,reward)
            episode_reward.append(reward)
            num_steps += 1

        t = time.time() - start_time

        result = [sum(episode_reward),episode_reward,num_steps,seed,sample_id,t] # total reward, reward per step, num_steps, seed, sample_id, time
        queue.put(result)

    except Exception as e:
        logger.error(f"Error in running experiment: {e}", exc_info=True)
        print(f"Error in running experiment: {e}")
        return


    return sum(episode_reward), episode_reward


def write_results_to_file(q,outfile,logger):
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



def run_experiment(config,make_env,agent):
    """Run an experiment with a given configuration
    

    Args:
        config (dict): Configuration dictionary.
        make_env (function): Function to create the environment. Takes in config and returns the non-stationary environment.
    """


    num_experiments= config['num_exp']
    results_dir = config['results_dir']
    experiment_name = config['experiment_name']
    logsdir = config['logs_dir']

    sample_id= [x for x in range(num_experiments)]


    log_name = experiment_name + ".log"
    os.makedirs(logsdir,exist_ok=True)
    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(logsdir,log_name), mode='w')])
    logger = logging.getLogger()


    manager = multiprocessing.Manager()
    queue = manager.Queue()
    

    os.makedirs(results_dir,exist_ok=True)

    
    seeds = random.sample(range(100000), num_experiments)


    outfile = os.path.join(results_dir,experiment_name + ".csv")

    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["total reward", "reward per step", "num_steps", "seed", "sample_id", "time"])

    writer_process = multiprocessing.Process(target=write_results_to_file, args=(queue,outfile,logger))
    writer_process.start()


    env = make_env(config)

    parameter_combinations = itertools.product(sample_id)
    input = [(queue,env,agent,seeds[i],sample_id[i],config,logger) for i,params in enumerate(parameter_combinations)]

    with Pool(config["num_workers"]) as p:
        p.starmap(run_episode, input)

    print("number of experiments",num_experiments)
    queue.put("DONE")
    writer_process.join()





if __name__ == "__main__":
    pass