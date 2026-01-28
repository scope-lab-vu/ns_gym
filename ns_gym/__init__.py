from gymnasium.envs.registration import register
from . import base  
from . import wrappers
from . import update_functions
from . import schedulers
from . import evaluate
from . import benchmark_algorithms
from . import context_switching
from . import utils
from . import envs

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ns_gym")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0-unknown"

register(
    id='ns_gym/Bridge-v0',
    entry_point='ns_gym.envs:Bridge',
    max_episode_steps=100,
)

register(
    id='ns_gym/VehicleTracking-v0',
    entry_point='ns_gym.envs.vehicle_tracking:VehicleTrackingEnv',
    max_episode_steps=100,

)

del register

__all__ = [
    "base", "wrappers", "update_functions", "schedulers", 
    "algo_utils", "benchmark_algorithms", "envs", "evaluate","utils"
]

# __all__ = ["base","wrappers", "update_functions", "schedulers", "algo_utils","benchmark_algorithms","envs","evaluate"]

# from .  import wrappers, update_functions, schedulers,evaluate,benchmark_algorithms, context_switching,evaluate



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