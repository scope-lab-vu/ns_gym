import ns_gym
import random
import os
import csv
import multiprocessing
from multiprocessing import Pool
import logging
import itertools
import time
import pandas as pd
import numpy as np
import torch
import ns_gym.utils

"""
Generic experiment runner functions
- Reads a YAML config file or command line arguments
- Builds non-stationary env
- Configures benchmark algorithms
- Stores results csv, trace visualizer
"""


class Queue:
    def __init__(self):
        self.queue = []

    def put(self, item):
        self.queue.append(item)

    def get(self):
        return self.queue.pop(0)

    def empty(self):
        return len(self.queue) == 0


def read_experiment_results(file_path):
    """Read the results of an experiment from a file.
    Args:
        file_path (str): Path to the file containing the results.
    Returns:
        pd.Dataframe: A pandas dataframe containing the results.
    """

    df = pd.read_csv(file_path)

    df["State-Action-Reward-NextState"] = df["State-Action-Reward-NextState"].apply(
        lambda x: eval(x)
    )

    return df


def array_to_list_if_array(x):
    """
    Converts a NumPy array to a Python list if the input is a NumPy array.
    If the array has a single element, return that value instead of a list.
    Otherwise, returns the input unchanged.

    Args:
        x: Input object to check and possibly convert.

    Returns:
        list, single value, or original object.
    """
    if isinstance(x, np.ndarray):
        if x.size == 1:  # If array has only one element
            return x.item()  # Return the single value
        return x.tolist()  # Otherwise, return the array as a list

    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.tolist()
    return x


def action_type_checker(action):
    if isinstance(action, torch.Tensor):
        return action.numpy()

    if isinstance(action, np.ndarray):
        return action


# def make_env(config):
#     raise NotImplementedError("make_env function not implemented")


def run_episode(queue, env, agent, seed, sample_id, config, logger):
    """Run an episode with a given agent and environment."""

    try:
        logger.info(f"Running experiment with seed {seed}")
        done = False
        truncated = False
        obs, _ = env.reset(seed=seed)
        obs, _ = ns_gym.utils.type_mismatch_checker(obs, None)

        episode_reward = []

        SARNS = []  # State, Action, Reward, Next State

        num_steps = 0
        start_time = time.time()

        while not done and not truncated:
            obs = ns_gym.utils.neural_network_checker(config["device"], obs)
            action = agent.act(obs)

            action = action_type_checker(action)

            next_obs, reward, done, truncated, info = env.step(action)
            next_obs, reward = ns_gym.utils.type_mismatch_checker(next_obs, reward)
            SARNS.append(
                (
                    array_to_list_if_array(obs),
                    array_to_list_if_array(action),
                    array_to_list_if_array(reward),
                    array_to_list_if_array(next_obs),
                )
            )
            obs = next_obs
            episode_reward.append(reward)
            num_steps += 1
            if num_steps == config["max_steps"] + 1:
                print("Max steps reached")
                break

        t = time.time() - start_time

        result = [
            sum(episode_reward),
            SARNS,
            num_steps,
            seed,
            sample_id,
            t,
        ]  # total reward, reward per step, num_steps, seed, sample_id, time
        queue.put(result)

    except Exception as e:
        logger.error(f"Error in running experiment: {e}", exc_info=True)
        print(f"Error in running experiment: {e}")
        return

    return sum(episode_reward), episode_reward


def write_results_to_file(q, outfile, logger):
    """Write results to file.
    Args:
        q (multiprocessing.Queue): Queue containing the results.
        outfile (str): Path to the output file.
    """
    try:
        with open(outfile, mode="a", newline="") as file:
            writer = csv.writer(file)
            while True:
                result = q.get()
                if result == "DONE":  # Check for sentinel value
                    break
                writer.writerow(result)
        logger.info("Results written to file")
    except Exception as e:
        print(f"Error in writing results to file: {e}")


def run_experiment(config, make_env, agent, multiprocess=True):
    """Run an experiment with a given configuration


    Args:
        config (dict): Configuration dictionary.
        make_env (function): Function to create the environment. Takes in config and returns the non-stationary environment.
    """

    num_experiments = config["num_exp"]
    results_dir = config["results_dir"]
    experiment_name = config["experiment_name"]
    logsdir = config["logs_dir"]

    sample_id = [x for x in range(num_experiments)]

    # Set up logging
    log_name = experiment_name + ".log"
    os.makedirs(logsdir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(logsdir, log_name), mode="w")],
    )
    logger = logging.getLogger()

    # Set up results directory
    os.makedirs(results_dir, exist_ok=True)

    # Set up random seeds
    seeds = random.sample(range(100000), num_experiments)

    # outfile
    outfile = os.path.join(results_dir, experiment_name + ".csv")

    with open(outfile, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "total_reward",
                "State-Action-Reward-NextState",
                "num_steps",
                "seed",
                "sample_id",
                "time",
            ]
        )

    if multiprocess:
        multiprocessing.set_start_method("spawn")
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        writer_process = multiprocessing.Process(
            target=write_results_to_file, args=(queue, outfile, logger)
        )
        writer_process.start()

        env = make_env(config)

        parameter_combinations = itertools.product(sample_id)
        input = [
            (queue, env, agent, seeds[i], sample_id[i], config, logger)
            for i, params in enumerate(parameter_combinations)
        ]

        print("Running experiments")
        with Pool(config["num_workers"]) as p:
            p.starmap(run_episode, input)

        print("Number of experiments", num_experiments)
        queue.put("DONE")
        writer_process.join()
        print("Results written to file")

        df = read_experiment_results(outfile)

        mean_reward = df["total_reward"].mean()
        std_err = df["total_reward"].std() / np.sqrt(num_experiments)
        print(f"Mean reward: {mean_reward}  +/- {std_err}")

    else:
        queue = Queue()

        print("Running experiments")
        for i in range(num_experiments):
            run_episode(
                queue, make_env(config), agent, seeds[i], sample_id[i], config, logger
            )
        print("Number of experiments", num_experiments)

        with open(outfile, mode="a", newline="") as file:
            writer = csv.writer(file)
            while not queue.empty():
                result = queue.get()
                writer.writerow(result)

        print("Results written to file")

        df = read_experiment_results(outfile)

        mean_reward = df["total_reward"].mean()
        std_err = df["total_reward"].std() / np.sqrt(num_experiments)
        print(f"Mean reward: {mean_reward}  +/- {std_err}")


if __name__ == "__main__":
    pass
