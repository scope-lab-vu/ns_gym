import itertools
import numpy as np
import ns_gym
import gymnasium as gym
from ns_gym.benchmark_algorithms import PPO, PPOActor, PPOCritic
import yaml
import concurrent.futures


def read_yaml_to_dict(file_path):
    """
    Reads a YAML file and converts it to a dictionary.

    Parameters:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary representation of the YAML content.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def main(config):
    env = gym.make(config["env_name"])
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    hidden_size = config["hidden_size"]

    actor = PPOActor(s_dim, a_dim, hidden_size)
    critic = PPOCritic(s_dim, hidden_size)

    ppo_agent = PPO(
        actor, 
        critic, 
        lr_policy=config["lr_policy"], 
        lr_critic=config["lr_critic"], 
        max_grad_norm=config["max_grad_norm"], 
        ent_weight=config["ent_weight"], 
        clip_val=config["clip_val"], 
        sample_n_epoch=config["n_epochs"], 
        sample_mb_size=config["minibatch_size"], 
        device=config["device"]
    )
    
    # Train the agent and return performance metric (e.g., total reward)
    result = ppo_agent.train_ppo(env, config)
    return result


def parallel_grid_search(config, param_grid, num_workers=1):
    """Perform parallel grid search for hyperparameter tuning.
    
    Args:
        config (dict): Base configuration dictionary.
        param_grid (dict): Dictionary defining hyperparameter grid with list of values.
        num_workers (int): Number of parallel workers.
    
    Returns:
        dict: Best configuration and its performance.
    """
    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_config = None
    best_performance = -np.inf

    def evaluate_combination(params):
        trial_config = config.copy()
        trial_config.update(params)
        print(f"Testing config: {trial_config}")
        return trial_config, main(trial_config)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_combination, params): params for params in param_combinations}

        for future in concurrent.futures.as_completed(futures):
            try:
                trial_config, performance = future.result()
                print(f"Config: {trial_config}, Performance: {performance}")

                if performance > best_performance:
                    best_performance = performance
                    best_config = trial_config
            except Exception as e:
                print(f"Error during evaluation: {e}")

    return {"best_config": best_config, "best_performance": best_performance}


if __name__ == "__main__":
    import argparse

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparameter_grid_config", help="Path to hyperparameter config file", required=True)
    parser.add_argument("--default_training_config", help="Path to default training parameters", required=True)
    parser.add_argument("--num_workers", type=int, help="Number of workers for parallel grid search", default=2)
    args = parser.parse_args()

    # Load configurations
    config = ns_gym.parse_config(args.default_training_config)
    param_grid = read_yaml_to_dict(args.hyperparameter_grid_config)

    # Run parallel grid search
    results = parallel_grid_search(config, param_grid, num_workers=args.num_workers)

    print("Best Configuration:", results["best_config"])
    print("Best Performance:", results["best_performance"])
