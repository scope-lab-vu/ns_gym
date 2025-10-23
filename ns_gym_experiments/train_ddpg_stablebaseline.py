import gymnasium as gym
import ns_gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def main():
    
    #env = gym.make(config["env_name"])
    env = gym.make("MountainCarContinuous-v0")

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),sigma=0.5)

    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(a_dim), sigma=float(config["noise_stddev"]) * np.ones(a_dim))

    model = DDPG(policy="MlpPolicy", env=env, action_noise=action_noise, verbose=1,gamma=0.98,learning_starts=10000,learning_rate=1e-3,buffer_size=200000,policy_kwargs=dict(net_arch=[400,300]))
    model.learn(300000)

    model.save("/media/--/home/n--/ns_gym_experiments/models/ddpg_mountaincar/ddpg_mountiancar_default")
    
if __name__ == "__main__":
    main()