import importlib as _importlib

from .algo_utils import *

# Lazy-loaded algorithm mapping: name -> (module_path, attribute_name)
_LAZY_IMPORTS = {
    "MCTS": (".MCTS", "MCTS"),
    "DQNAgent": (".DDQN.DDQN", "DQNAgent"),
    "DQN": (".DDQN.DDQN", "DQN"),
    "train_ddqn": (".DDQN.DDQN", "train_ddqn"),
    "PAMCTS": (".PAMCTS", "PAMCTS"),
    "AlphaZeroAgent": (".AlphaZero.alphazero", "AlphaZeroAgent"),
    "AlphaZeroNetwork": (".AlphaZero.alphazero", "AlphaZeroNetwork"),
    "PPO": (".PPO.PPO", "PPO"),
    "PPOActor": (".PPO.PPO", "PPOActor"),
    "PPOCritic": (".PPO.PPO", "PPOCritic"),
    "DDPG": (".DDPG", "DDPG"),
}

__all__ = [
    "MCTS",
    "DQN",
    "DQNAgent",
    "train_ddqn",
    "PAMCTS",
    "AlphaZeroAgent",
    "AlphaZeroNetwork",
    "PPO",
    "PPOActor",
    "PPOCritic",
    "DDPG",
]


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
