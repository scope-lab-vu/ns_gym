# from typing import Any, SupportsFloat, Type
from typing import Any
import gymnasium as gym

# import gymnasium.spaces as spaces
# import numpy as np
from gymnasium import Wrapper


"""
Wrappers for the box2d environments.
"""


class NSBox2dWrapper(Wrapper):
    """Wrapper for Box2d envs, TODO!"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        raise NotImplementedError

    def step(self, action: Any):
        pass

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def close(self):
        return super().close()

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()
