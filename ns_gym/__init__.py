import importlib as _importlib

from gymnasium.envs.registration import register

# Eager imports: core modules that are lightweight and always needed
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

# Lazy imports: heavy submodules loaded on first access
_LAZY_SUBMODULES = {
    "benchmark_algorithms",
    "context_switching",
    "evaluate",
}

__all__ = [
    "base", "wrappers", "update_functions", "schedulers",
    "benchmark_algorithms", "envs", "evaluate", "utils",
    "context_switching",
]


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
