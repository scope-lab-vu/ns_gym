import numpy as np


def discritize_state_space(k):
    """Discritize the state space of the mountain car environment.

    The mountain car state space is the the x-axis position and the velocity of the car.
    """

    pos_low = -1.2
    pos_high = 0.6 
    vel_low = -0.07
    vel_high = 0.07
    

    pos_bins = np.linspace(pos_low, pos_high, k)
    vel_bins = np.linspace(vel_low, vel_high, k)

    return pos_bins, vel_bins













if __name__ == "__main__":
    import gymnasium as gym
    import ns_gym as nsb

    k = 25
    pos_bins, vel_bins = discritize_state_space(k)

    print(pos_bins)
    print(vel_bins)
    print("--------------------")

    env = gym.make("MountainCar-v0")

    obs,_ = env.reset()

    print(obs)

    pos, vel = obs

    closest_pos = min(pos_bins, key=lambda x:abs(x-pos))
    closest_vel = min(vel_bins, key=lambda x:abs(x-vel))
     
    print(closest_pos)
    print(closest_vel)
    print("--------------------")

    
    