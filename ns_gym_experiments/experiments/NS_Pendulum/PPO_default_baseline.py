import ns_gym 
from ns_gym.eval import run_experiment
from ns_gym.utils import parse_config   
from ns_gym.benchmark_algorithms import PPO,PPOActor, PPOCritic
import gymnasium as gym

import torch



def make_env(config):
    env = gym.make(config["env_name"])
    return env

def make_agent(config):
    s_dim = config["s_dim"]
    a_dim = config["a_dim"]
    hidden_size = config["hidden_size"]

    device = config["device"]

    actor = PPOActor(s_dim,a_dim,hidden_size)
    critic = PPOCritic(s_dim,hidden_size)


    actor.to(device)
    actor.load_state_dict(torch.load(config["actor_weights_path"],weights_only=True,map_location=device))

    agent = PPO(actor, 
                critic, 
                lr_policy=config["lr_policy"], 
                lr_critic=config["lr_critic"], 
                max_grad_norm=config["max_grad_norm"], 
                ent_weight=config["ent_weight"], 
                clip_val=config["clip_val"], 
                sample_n_epoch=config["sample_n_epoch"], 
                sample_mb_size=config["sample_mb_size"], 
                device=device)

    return agent


def main(config,make_env):
    agent = make_agent(config)

    print("Running experiment")
    print(f"Config: {config}  \n")
    run_experiment(config, make_env,agent)
    print("Experiment complete")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run PPO on Pendulum')
    parser.add_argument('--config', type=str, default="ns_gym_experiments/configs/ppo_pendulum_eval.yaml")

    args = parser.parse_args()
    config_path = args.config

    config = parse_config(config_path)

    main(config,make_env)
    
    
    