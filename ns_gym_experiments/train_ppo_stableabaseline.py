import gymnasium as gym
import ns_gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



def main():
    
    #env = gym.make(config["env_name"])
    #env = gym.make("MountainCarContinuous-v0")
    # vec_env = make_vec_env("MountainCarContinuous-v0",n_envs=1)
    # vec_env = make_vec_env("Acrobot-v1",n_envs=16)

    env = gym.make("Pendulum-v1")
    vec_env = make_vec_env("Pendulum-v1",n_envs=4)

    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(a_dim), sigma=float(config["noise_stddev"]) * np.ones(a_dim))
    model = PPO(policy="MlpPolicy",env=vec_env,n_steps=1024,gae_lambda=0.95, gamma=0.9, n_epochs=10,ent_coef=0.0, learning_rate=1e-3,clip_range=0.2,use_sde=True,sde_sample_freq=4)

    model.learn(1e5)

    model.save("/Users/--/Documents/--/Research/ns_gym_project/ns_gym_experiments/models/ppo_pendulum")
    
if __name__ == "__main__":
    main()

