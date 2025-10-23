import gymnasium as gym
import ns_gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise


def main():
    
    #env = gym.make(config["env_name"])
    #env = gym.make("MountainCarContinuous-v0")
    # vec_env = make_vec_env("MountainCarContinuous-v0",n_envs=1)
    # vec_env = make_vec_env("Acrobot-v1",n_envs=16)

    env = gym.make("Pendulum-v1")
    vec_env = make_vec_env("Pendulum-v1",n_envs=4)

    action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))

    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(a_dim), sigma=float(config["noise_stddev"]) * np.ones(a_dim))
    model = TD3("MlpPolicy",env=vec_env,gamma=0.98,buffer_size=20000,learning_starts=10000,action_noise=action_noise,gradient_steps=1,train_freq=1,learning_rate=1e-3,policy_kwargs=dict(net_arch=[400, 300]))
    model.learn(20000)

    model.save("/Users/--/Documents/--/Research/ns_gym_project/ns_gym_experiments/models/td3_pendulum/td3_pendulum")
    
if __name__ == "__main__":
    main()

