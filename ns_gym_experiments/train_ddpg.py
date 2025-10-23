import numpy as np
import ns_gym
import gymnasium as gym
from ns_gym.benchmark_algorithms import DDPG

def main(config):

    env = gym.make(config["env_name"])
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    hidden_size = config["hidden_size"]
    num_episodes = config["num_episodes"]


    ddpg_agent = DDPG(s_dim, a_dim, hidden_size)

    batch_size = config["batch_size"]   
    gamma = config["gamma"]
    tau = config["tau"]
    warmup_episodes = config["warmup_episodes"]
    save_path = config["save_path"] # path to directory where model will be saved (Actor and Critic)

    print("Training DDPG on ", config["env_name"])
    best_score = ddpg_agent.train(env,num_episodes,batch_size,gamma,tau,warmup_episodes,save_path)
    print("Training complete. Best score: ", best_score)


if __name__ == "__main__":
    config = ns_gym.parse_config("configs/ppo_pendulum_training.yaml")
    main(config)