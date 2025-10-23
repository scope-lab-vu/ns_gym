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


    max_episodes = config["max_episodes"]
    batch_size = config["batch_size"]
    minibatch_size = config["minibatch_size"]
    n_epochs = config["n_epochs"]
    hidden_size = config["hidden_size"] # was 64
    max_steps = config["max_steps"]
    gamma = config["gamma"]
    lamb = config["lamb"]
    device = config["device"]



    lr_policy = config["lr_policy"]
    lr_critic = config["lr_critic"]
    max_grad_norm = config["max_grad_norm"]
    clip_val = config["clip_val"]
    ent_weight = config["ent_weight"]


    ppo_agent = PPO(actor, critic, lr_policy=lr_policy, lr_critic=lr_critic, max_grad_norm=max_grad_norm, 
                ent_weight=ent_weight, clip_val=clip_val, sample_n_epoch=n_epochs, sample_mb_size=minibatch_size, device=device)
    

    ppo_agent.train_ppo(env,config)

if __name__ == "__main__":
    config = ns_gym.parse_config("configs/ppo_mountaincar_training.yaml")
    main(config)