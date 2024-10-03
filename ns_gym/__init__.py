from gymnasium.envs.registration import register

register(
    id='ns_gym/Bridge-v0',
    entry_point='ns_gym.envs:Bridge',
    max_episode_steps=100,
)


__all__ = ["base","wrappers", "update_functions", "schedulers", "utils","benchmark_algorithms","envs"]

from .wrappers import *
from .update_functions import *
from .schedulers import *
from .utils import *
from .benchmark_algorithms import *
from ns_gym.envs.Bridge import Bridge
