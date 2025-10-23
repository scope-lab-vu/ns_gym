import gymnasium as gym
import ns_gym
import csv
import logging
import yaml

def make_contin_env(wrapper_config):
    change_notification = wrapper_config["change_notification"]
    delta_change_notification = wrapper_config["delta_change_notification"]
    initial_prob_dist = wrapper_config["initial_prob_dist"]

    scheduler = ns_gym.schedulers.ContinuousScheduler()
    updateFn = ns_gym.update_functions.DistributionDecrmentUpdate(scheduler,k=0.1)

    tunable_params = {"P":updateFn}
    env = gym.make("FrozenLake-v1",max_episode_steps=100)
    realworld_env = ns_gym.wrappers.NSFrozenLakeWrapper(env,
                                                      tunable_params=tunable_params,
                                                      change_notification=change_notification,
                                                      delta_change_notification=delta_change_notification,
                                                      initial_prob_dist=initial_prob_dist)

    return realworld_env

def make_discrete_env(p,wrapper_config):
    change_notification = wrapper_config["change_notification"]
    delta_change_notification = wrapper_config["delta_change_notification"]
    initial_prob_dist = wrapper_config["initial_prob_dist"]

    scheduler = ns_gym.schedulers.DiscreteScheduler({0})
    updateFn = ns_gym.update_functions.DistributionStepWiseUpdate(scheduler,update_values=[[p,(1-p)/2,(1-p)/2]])
    params = {'P':updateFn}

    tunable_params = {"P":updateFn}
    env = gym.make("FrozenLake-v1",max_episode_steps=100)
    realworld_env = ns_gym.wrappers.NSFrozenLakeWrapper(env,
                                                      tunable_params=tunable_params,
                                                      change_notification=change_notification,
                                                      delta_change_notification=delta_change_notification,
                                                      initial_prob_dist=initial_prob_dist)

    return realworld_env


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

def read_config_file(config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config["gym_env"],config["wrapper"],config["agent"],config["experiment"]
