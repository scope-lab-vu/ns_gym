import ns_gym
from ns_gym.base import Observation,Reward
import gymnasium as gym
from gymnasium import spaces
from gymnasium import ObservationWrapper, RewardWrapper
from gymnasium.vector import SyncVectorEnv,AsyncVectorEnv
from ns_gym.wrappers import NSFrozenLakeWrapper
from ns_gym.schedulers import ContinuousScheduler,DiscreteScheduler
from ns_gym.update_functions import DistributionDecrementUpdate
import numpy as np

import time
from ns_gym.benchmark_algorithms import MCTS


from typing import Any

'''
NOTE: While we can hack through some basic syncchnous vectorization the hard part will be hadeling the get_planning env module. 
Perhaps that doesnt matter though because integrating vectorized envornments in tree search can be tricky. IDK any frameworks that do this out of the box (perhaps lightzero...)
'''


CHANGE_KEYS = ["P", "R", "T"]  # list all possible keys ahead of time

class FlattenObsWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=0, high=env.observation_space.n - 1, shape=(), dtype=np.int64),
            "env_change": spaces.Box(low=0.0, high=1.0, shape=(len(CHANGE_KEYS),), dtype=np.float32),
            "delta_change": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "relative_time": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        })
    def observation(self, obs: Observation) -> dict:
        print(obs)
        # Always return a NumPy array for env_change_flags.
        if obs.env_change is None:
            env_change_flags = np.array([0.0 for _ in CHANGE_KEYS], dtype=np.float32)

        elif obs.env_change is True:
            env_change_flags = np.array([1.0 for _ in CHANGE_KEYS], dtype=np.float32)
        else:
            env_change_flags = np.array(
                [obs.env_change.get(k, False) for k in CHANGE_KEYS],
                dtype=np.float32
            )
        return {
            "state": np.array(obs.state) if not isinstance(obs.state, np.ndarray) else obs.state,
            "env_change": env_change_flags,
            "delta_change": float(obs.delta_change) if obs.delta_change is not None else 0.0,
            "relative_time": float(obs.relative_time),
        }


class FlattenRewardWrapper(RewardWrapper):
    def reward(self, rew: Reward) -> float:
        return float(rew.reward)
    

env = gym.make('FrozenLake-v1',render_mode="rgb_array",is_slippery=False)

### Define the scheduler #######
scheduler = ContinuousScheduler() #Update the slipperiness at each timestep

#### Define the update function #####
update_function = DistributionDecrementUpdate(scheduler=scheduler,k = 0.1) #Decrement the slipperiness by 0.1 at each timestep where the scheduler fires true

# Map parameter to update function
####### Defining environmental parameters ############
param = "P"
params = {param:update_function}
###### Import the frozen lake
ns_env = NSFrozenLakeWrapper(env,params, initial_prob_dist=[1,0,0])
ns_env = FlattenObsWrapper(ns_env)
ns_env = FlattenRewardWrapper(ns_env)


def make_env():
    def _init():
        env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=False)
        scheduler = ContinuousScheduler()
        update_function = DistributionDecrementUpdate(scheduler=scheduler, k=0.1)
        param = "P"
        params = {param: update_function}
        ns_env = NSFrozenLakeWrapper(env, params, initial_prob_dist=[1, 0, 0],delta_change_notification=True, change_notification=True)
        ns_env = FlattenObsWrapper(ns_env)     # <-- apply wrapper here 
        ns_env = FlattenRewardWrapper(ns_env)  # <-- and here
        return ns_env
    return _init

if __name__ == "__main__":
    vec_env = SyncVectorEnv([make_env() for _ in range(4)])
    out = vec_env.reset()
    print(out)

    vec_env.step(actions=np.array([0 for x in range(4)]))








