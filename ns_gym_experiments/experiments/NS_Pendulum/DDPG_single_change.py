import gymnasium as gym
import ns_gym



######### Import make agent and main from the DDPG_default_baseline.py file #########

from DDPG_default_baseline import make_agent, main

###############


def make_env(config):
    """
    Increase the mass of the pendulum by 50%.
    """
    env = gym.make(config["env_name"])
    tunable_param = config["tunable_params"]

    scheduler = ns_gym.schedulers.ContinuousScheduler(start=0,end=0)
    update_fn = ns_gym.update_functions.IncrementUpdate(scheduler,1)
    param_map  = {tunable_param[0]:update_fn}
    ns_env = ns_gym.wrappers.NSClassicControlWrapper(env,param_map)
    return ns_env


if __name__ == "__main__":
    import argparse
    from ns_gym.utils import parse_config

    # parser = argparse.ArgumentParser(description='Run PPO on Pendulum')
    # parser.add_argument('--config', type=str, default="ns_gym_experiments/configs/ppo_pendulum_eval.yaml")

    # args = parser.parse_args()
    # config_path = args.config

    config_path = input("Enter the path to the config file: ")

    config = parse_config(config_path)

    save_path = input("Enter the path to save the results: ")
    exp_name = input("Enter the experiment name: ")

    config["results_dir"] = save_path
    config["experiment_name"] = exp_name

    main(config,make_env)