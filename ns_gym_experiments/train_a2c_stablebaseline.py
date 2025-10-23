import gymnasium as gym
import ns_gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env



def main():
    
    #env = gym.make(config["env_name"])
    #env = gym.make("MountainCarContinuous-v0")
    # vec_env = make_vec_env("MountainCarContinuous-v0",n_envs=1)
    # vec_env = make_vec_env("Acrobot-v1",n_envs=16)

    env = gym.make("Acrobot-v1")
    vec_env = make_vec_env("Acrobot-v1",n_envs=16)

    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(a_dim), sigma=float(config["noise_stddev"]) * np.ones(a_dim))
    model = A2C("MlpPolicy",env=vec_env,ent_coef=0.0,verbose=1)

    model.learn(5e5)

    model.save("/Users/--/Documents/--/Research/ns_gym_project/ns_gym_experiments/models/a2c_acrobot/a2c_acrobot")
    
if __name__ == "__main__":
    main()

