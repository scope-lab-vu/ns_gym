import itertools
import numpy as np
import ns_gym
import gymnasium as gym
from ns_gym.benchmark_algorithms import PPO, PPOActor, PPOCritic


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


def grid_search(config, param_grid):
    """Perform grid search for hyperparameter tuning.
    
    Args:
        config (dict): Base configuration dictionary.
        param_grid (dict): Dictionary defining hyperparameter grid with list of values.
    
    Returns:
        dict: Best configuration and its performance.
    """
    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_config = None
    best_performance = -np.inf

    for i, params in enumerate(param_combinations):
        # Merge base config with current hyperparameter combination
        trial_config = config.copy()
        trial_config.update(params)

        print(f"Trial {i + 1}/{len(param_combinations)}: Testing config {trial_config}")
        performance = main(trial_config)
        print(f"Performance: {performance}")

        # Update the best configuration if necessary
        if performance > best_performance:
            best_performance = performance
            best_config = trial_config

    return {"best_config": best_config, "best_performance": best_performance}


if __name__ == "__main__":
    # Load base config
    config = ns_gym.parse_config("configs/ppo_acrobot_training.yaml")

    # Define hyperparameter grid
    param_grid = {
        "hidden_size": [32, 64, 128],
        "lr_policy": [1e-4, 1e-3, 1e-2],
        "lr_critic": [1e-4, 1e-3, 1e-2],
        "gamma": [0.9, 0.95, 0.99],
        "clip_val": [0.1, 0.2, 0.3],
    }

    # Run grid search
    results = grid_search(config, param_grid)

    print("Best Configuration:", results["best_config"])
    print("Best Performance:", results["best_performance"])
