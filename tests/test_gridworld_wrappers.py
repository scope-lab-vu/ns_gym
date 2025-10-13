import pytest
import gymnasium as gym
import ns_gym
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import IncrementUpdate, RandomWalk
import numpy as np
import copy


SUPPORTED_GRID_WORLD_ENV_IDS = [
    "CliffWalking-v1",
    "FrozenLake-v1",
    "ns_gym/Bridge-v0"
]


def test_cliffwalking_wrapper():
    env = gym.make("CliffWalking-v1")
    env_name = env.unwrapped.__class__.__name__

    from ns_gym.wrappers import NSCliffWalkingWrapper
    assert NSCliffWalkingWrapper is not None

    tunable_params = list(ns_gym.base.TUNABLE_PARAMS[env_name].keys())
    param_map = {param: ns_gym.update_functions.DistributionIncrementUpdate(ns_gym.schedulers.ContinuousScheduler(), k=0.1) for param in tunable_params}
    ns_env = NSCliffWalkingWrapper(env, param_map, change_notification=True, delta_change_notification=True)

    assert isinstance(ns_env, NSCliffWalkingWrapper)



def test_frozenlake_wrapper():
    env = gym.make("FrozenLake-v1")

    from ns_gym.wrappers import NSFrozenLakeWrapper
    assert NSFrozenLakeWrapper is not None

    env_name = env.unwrapped.__class__.__name__
    tunable_params = list(ns_gym.base.TUNABLE_PARAMS[env_name].keys())

    param_map = {param: ns_gym.update_functions.DistributionIncrementUpdate(ns_gym.schedulers.ContinuousScheduler(), k=0.1) for param in tunable_params}
    # Placeholder: Implementation needed for NSFrozenLakeWrapper
    ns_env = NSFrozenLakeWrapper(env, param_map, change_notification=True, delta_change_notification=True)

    assert isinstance(ns_env, NSFrozenLakeWrapper)



