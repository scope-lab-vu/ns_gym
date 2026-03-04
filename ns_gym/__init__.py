from gymnasium.envs.registration import register
from . import base
from . import wrappers
from . import update_functions
from . import schedulers
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

# Lazy-loaded submodules: benchmark_algorithms, evaluate, context_switching
# These pull in heavy dependencies (torch, pandas, stable_baselines3, matplotlib)
# and are only loaded on first access via __getattr__.
_LAZY_SUBMODULES = {"evaluate", "benchmark_algorithms", "context_switching"}


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "base", "wrappers", "update_functions", "schedulers",
    "utils", "envs", "benchmark_algorithms", "evaluate", "context_switching",
]