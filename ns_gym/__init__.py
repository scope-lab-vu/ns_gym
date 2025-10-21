from gymnasium.envs.registration import register

register(
    id='ns_gym/Bridge-v0',
    entry_point='ns_gym.envs:Bridge',
    max_episode_steps=100,
)

del register

__all__ = ["base","wrappers", "update_functions", "schedulers", "algo_utils","benchmark_algorithms","envs","evaluate"]


from .  import wrappers, update_functions, schedulers,evaluate,benchmark_algorithms, context_switching,evaluate

__version__ = "0.0.2"


# import .utils
# import .benchmark_algorithms
# import .eval
# import context_switching


# from .wrappers import *
# from .update_functions import *
# from .schedulers import *
# from .utils import *
# from .benchmark_algorithms import *
# from ns_gym.envs.Bridge import Bridge
# from .eval import *
# from .context_switching import *