import gymnasium as gym
import ns_gym
from ns_gym.base import UpdateFn, UpdateDistributionFn


class PursuitEvasionWrapper(ns_gym.base.NSWrapper):
    """Wrapper to adapt CityEnvGym's Pursuit-Evasion environment to the ns_gym interface."""

    def __init__(
        self,
        env: gym.Env,
        tunable_params: dict[str, type[UpdateFn] | type[UpdateDistributionFn]],
        change_notification: bool = False,
        delta_change_notification: bool = False,
        in_sim_change: bool = False,
        **kwargs,
    ):
        super().__init__(
            env,
            tunable_params,
            change_notification,
            delta_change_notification,
            in_sim_change,
            **kwargs,
        )

    def step(self, action, env_change: dict[str, bool], delta_change: dict[str, bool]):
        return super().step(action, env_change, delta_change)

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def get_planning_env(self):
        raise NotImplementedError(
            "get_planning_env is not implemented for PursuitEvasionWrapper."
        )

    def __deepcopy__(self, memo):
        raise NotImplementedError(
            "__deepcopy__ is not implemented for PursuitEvasionWrapper."
        )

    def render(self):
        return super().render()
