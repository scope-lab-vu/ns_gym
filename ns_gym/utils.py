# import ns_bench.base as base
import math
import numpy as np
from scipy.stats import wasserstein_distance
import ns_gym.base as base
import yaml
import argparse
from pathlib import Path
import datetime
import warnings
import torch


def state_action_update(transitions: list, new_probs: list):
    """Update the transitions associated with a single state action pair in the gridworld environments with new probs

    Args:
        transitions (list): A list of tuples representing the transitions for a specific state-action pair.
        new_probs (list): A list of new probabilities to update the transitions with.

    Returns:
        list: The updated list of transitions with the new probabilities.

    Notes:
        The possible transitions from state s with action a are typically stored in a list of tuples.
        For example, the possible transitions for (s,a) in FrozenLake are store in table P at P[s][a],
        where:

        P[s][a] = [(p0, newstate0, reward0, terminated),
                        (p1, newstate1, reward1, terminated),
                        (p2, newstate2, reward2, terminated)]

        The indended direction is stored in P[s][a][1].
    """

    for i, tup in enumerate(transitions):
        temp = list(tup)
        temp[0] = new_probs[i]
        transitions[i] = tuple(temp)
    return transitions


def n_choose_k(n, k):
    """Calculate the binomial coefficient 'n choose k'
    Args:
        n (int): Total number of items.
        k (int): Number of items to choose.

    Returns:
        int: The binomial coefficient, i.e., the number of ways to choose k items
            from a set of n items.
    """

    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def wasserstein_dual(u: np.ndarray, v: np.ndarray):
    """_summary_

    Args:
        u (np.ndarray): _description_
        v (np.ndarray): _description_
        d (np.ndarray): _description_

    Returns:
        _type_: _description_
    """

    dist = wasserstein_distance(u, v)
    return dist


def categorical_sample(probs: list):
    """Sample from a categorical distribution
    Args:
        probs (list): A list of probabilities that sum to 1
    Returns:
        int: The index of the sampled probability
    """
    return np.random.choice(len(probs), p=probs)


# def update_frozen_lake_table(P,nS,nA):
#     """Update the transition table for the FrozenLake environment
#     Args:
#         P : probability table for the FrozenLake environment. P[s][a] = [(p0, newstate0, reward0, terminated),
#                         (p1, newstate1, reward1, terminated),
#                         (p2, newstate2, reward2, terminated)
#     Returns:
#         _type_: _description_
#     """
#     for s in range(nS):
#         for a in range(nA):
#             transitions = P[s][a]
#             for i


def type_mismatch_checker(observation=None, reward=None):
    """
    A helper function to handle type mismatches between ns_gym
    and Gymnasium environments.

    Args:
        observation: The observation, which may be an dictionary.
        reward: The reward object, which may be an instance of nsg.base.Reward.

    Returns:
        A tuple containing the processed observation and reward, with None if not provided.
    """
    # Process the observation if provided
    if observation is not None:
        if isinstance(observation, dict) and "state" in observation:
            obs = observation["state"]
        else:
            obs = observation
    else:
        obs = None

    # Process the reward if provided
    if reward is not None:
        if isinstance(reward, base.Reward):
            rew = reward.reward
        else:
            rew = reward
    else:
        rew = None
    assert not isinstance(obs, dict), "Observation is still a dict after type checking."
    return obs, rew


def parse_config(file_path):
    """
    Reads a YAML config file and updates its values with any command-line arguments.
    Also checks if necessary configs for experiments are present.

    Args:
        file_path (str or Path): Path to the YAML configuration file.

    Returns:
        dict: The combined configuration dictionary.
    """
    # Ensure file_path is a Path object
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file '{file_path}' not found.")

    # Load configuration from the YAML file
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse command-line arguments to override config values
    parser = argparse.ArgumentParser(description="Override configuration parameters.")

    for key, value in config.items():
        # Dynamically add arguments based on the YAML keys
        arg_type = type(value) if value is not None else str
        parser.add_argument(
            f"--{key}", type=arg_type, help=f"Override {key} in config."
        )

    # Parse command-line arguments
    args = parser.parse_args()

    # Update the config with provided command-line arguments
    for key in config.keys():
        arg_value = getattr(args, key, None)
        if arg_value is not None:
            config[key] = arg_value

    # Check if necessary configs are present

    if "num_exp" not in config:
        config["num_exp"] = 1
        warnings.warn("num_exp not found in config. Defaulting to 1.")

    if "results_dir" not in config:
        # results_dir = pathlib.Path(__file__).parent / 'results'
        # config['results_dir'] = results_dir
        # os.makedirs(results_dir, exist_ok=True)
        # warnings.warn("results_dir not found in config. Defaulting to 'results'.")
        raise ValueError(
            "results_dir not found in config. Please specify a results directory."
        )

    if "experiment_name" not in config:
        config["experiment_name"] = (
            f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        warnings.warn(
            "experiment_name not found in config. Defaulting to current timestamp."
        )

    if "logs_dir" not in config:
        # logsdir = pathlib.Path(__file__).parent / 'logs'
        # config['logsdir'] = logsdir
        # os.makedirs(logsdir, exist_ok=True)
        # warnings.warn("logsdir not found in config. Defaulting to 'logs'.")
        raise ValueError(
            "logs_dir not found in config. Please specify a logs directory."
        )

    if "num_workers" not in config:
        config["num_workers"] = 1
        warnings.warn("num_workers not found in config. Defaulting to 1.")

    return config


def neural_network_checker(agent_device, obs):
    """
    Helper function to check if model and inputs are on the same device
    """

    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()

    obs = obs.to(agent_device)

    return obs


if __name__ == "__main__":
    test = n_choose_k(5, 2)
