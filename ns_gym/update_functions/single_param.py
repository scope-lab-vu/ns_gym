import ns_gym.base as base
from typing import Union, Any, Type
import numpy as np

"""
These are a collection of parameter update functions that can be used in the ns_bench framework.
These classes take in a single parameter (scalars) and return a new parameter. 
The update functions only "fire" when the scheduler returns True.

To maintain a consistent interface, the update functions can be implemented as a derived class of base.UpdateFn.

But really the only main requirement is that the update function takes in a parameter and a time step in the
__call__() method and returns a new parameter and a boolean value indicating whether the update function fired or not.

"""

##### Update Functions for single parameters (scalars) #####


class DeterministicTrend(base.UpdateFn):
    r"""Update the parameter with a deterministic trend.

    Overview:
        .. math::
            Y_t = Y_{t-1} + slope * t

        where :math:`Y_t` is the parameter value at time step :math:`t` and slope is the slope of the trend.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        slope (float): The slope of the trend.
    """

    def __init__(self, scheduler: Type[base.Scheduler], slope: float) -> None:
        super().__init__(scheduler)
        self.slope = slope

    def _update(self, param: float, t: float) -> tuple[float, bool]:
        updated_param = param + self.slope * t
        return updated_param


class RandomWalkWithDriftAndTrend(base.UpdateFn):
    r"""Parameter update function that updates the parameter with white noise and a deterministic trend.

    Overview:

        .. math::
            Y_t = \alpha + Y_{t-1} + \text{slope} * t + \epsilon_t

        where :math:`Y_t` is the parameter value at time step :math:`t`, :math:`\alpha` is the drift term, :math:`\text{slope}` is the slope of the trend, and :math:`\epsilon` is white noise.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        alpha (float): The drift term.
        mu (float): The mean of the white noise.
        sigma (float): The standard deviation of the white noise.
        slope (float): The slope of the trend.
        seed (Union[int, None], optional): Seed for the random number generator. Defaults to None.
    """

    def __init__(
        self,
        scheduler: Type[base.Scheduler],
        alpha: float,
        mu: float,
        sigma: float,
        slope: float,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.slope = slope
        self.rng = np.random.default_rng(seed=seed)

    def _update(self, param: float, t: float) -> tuple[float, bool]:
        white_noise = self.rng.normal(self.mu, self.sigma, 1)
        updated_param = self.alpha + param + white_noise + self.slope * t
        return updated_param


class RandomWalk(base.UpdateFn):
    r"""Parameter update function that updates the parameter with white noise.

    Overview:
        A pure random walk : :math:`Y_t = Y_{t-1} + \epsilon_t` where :math:`Y_t` is the parameter value at time step :math:`t`
        and :math:`\epsilon` is white noise.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        mu (Union[float,int], optional): The mean of the white noise. Defaults to 0.
        sigma (Union[float,int], optional): The standard deviation of the white noise. Defaults to 1.
        seed (Union[int,None], optional): Seed for the random number generator. Defaults to None.
    """

    def __init__(
        self,
        scheduler: Type[base.Scheduler],
        mu: Union[float, int] = 0,
        sigma: Union[float, int] = 1,
        seed=None,
    ) -> tuple[Any, bool]:
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed=seed)

    def _update(self, param: Any, t: Union[int, float]) -> Any:
        white_noise = self.rng.normal(self.mu, self.sigma, 1)
        updated_param = param + white_noise
        return updated_param[0]


class RandomWalkWithDrift(base.UpdateFn):
    r"""A parameter update function that updates the parameter with white noise and a drift term.

    Overview:

        .. math::
            Y_t = \alpha + Y_{t-1} + \epsilon_t

        where :math:`Y_t` is the parameter value at time step :math:`t`, :math:`\alpha` is the drift term, and :math:`\epsilon` is white noise.


    Args:
        alpha (float): The drift term.
        mu (float): The mean of the white noise.
        sigma (float): The standard deviation of the white noise.
        seed (int): Seed for the random number generator. Defaults to None.
    """

    def __init__(
        self,
        scheduler: Type[base.Scheduler],
        alpha: float,
        mu: float,
        sigma: float,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.rng = np.random.default_rng(seed=seed)

    def _update(self, param: Any, t: int) -> Any:
        white_noise = self.rng.normal(self.mu, self.sigma, 1)
        upated_param = self.alpha + param + white_noise
        return upated_param


class IncrementUpdate(base.UpdateFn):
    r"""Increment the the parameter by k.

    Overview:
        .. math::
            Y_t = Y_{t-1} + k

        where :math:`Y_t` is the parameter value at time step :math:`t` and :math:`k` is the amount to increment the parameter by.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        k (float): The amount which the parameter is updated.

    """

    def __init__(self, scheduler: Type[base.Scheduler], k: float) -> None:
        super().__init__(scheduler)
        self.k = k

    def _update(self, param: Any, t: int) -> Any:
        param += self.k
        return param


class DecrementUpdate(base.UpdateFn):
    r"""Decrement the probability of going in the intended direction by some k.

    Overview:

        .. math::
            Y_t = Y_{t-1} - k

        where :math:`Y_t` is the parameter value at time step :math:`t` and :math:`k` is the amount to decrement the parameter by.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        k (float): The amount which the parameter is updated.
    """

    def __init__(self, scheduler, k) -> None:
        super().__init__(scheduler)
        self.k = k

    def _update(self, param, t) -> Any:
        param -= self.k
        return param


class StepWiseUpdate(base.UpdateFn):
    r"""Update the parameter at specific time steps.

    Overview:
        This function updates the parameter to the next value in the `param_list` when called. If the `param_list` is empty, the parameter is not updated.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        param_list (list): A list of parameters to update to.
    """

    def __init__(self, scheduler: Type[base.Scheduler], param_list: list) -> None:
        super().__init__(scheduler)
        self.param_list = param_list

    def _update(self, param: list, t: int) -> Any:
        try:
            param = self.param_list.pop(0)
        except AssertionError:
            "No more parameters to update"
        finally:
            return param


class NoUpdate(base.UpdateFn):
    r"""Do not update the parameter but return correct interface

    Overview:
        This function does not update the parameter when called. It is useful for testing and debugging.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
    """

    def __init__(self, scheduler: Type[base.Scheduler]) -> None:
        super().__init__(scheduler)

    def _update(self, param: Any, t: int) -> Any:
        return param


class OscillatingUpdate(base.UpdateFn):
    r"""Update the parameter with an oscillating function.

    Overview:

        .. math::
            Y_t = Y_{t-1} + \delta * sin(t)

        where :math:`Y_t` is the parameter value at time step :math:`t` and :math:`\delta` is the amplitude of the sine wave.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        delta (float): The amplitude of the sine wave.
    """

    def __init__(self, scheduler: Type[base.Scheduler], delta: float) -> None:
        super().__init__(scheduler)
        self.delta = delta

    def _update(self, param: Any, t: int) -> Any:
        oscillation = self.delta * np.sin(t)
        return param + oscillation

class ExponentialDecay(base.UpdateFn):
    r"""Exponential decay of the parameter.
    
    Overview:

        .. math::
            Y_t = Y_0 * exp(-\lambda * t)

        where :math:`Y_t` is the parameter value at time step :math:`t`, :math:`Y_0` is the initial parameter value, and :math:`\lambda` is the rate of decay.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        decay_rate (float): The rate of decay. i.e. :math:`\lambda`
    """

    def __init__(self, scheduler: Type[base.Scheduler], decay_rate: float) -> None:
        super().__init__(scheduler)
        self.decay_rate = decay_rate

    def _update(self, param: Any, t: int) -> Any:
        updated_param = param * np.exp(-self.decay_rate * t)
        return updated_param


class GeometricProgression(base.UpdateFn):
    r"""Apply a geometric progression to the parameter.

    Overview:

        .. math::
            Y_t = Y_0 * r^t

        where :math:`Y_t` is the parameter value at time step :math:`t`, :math:`Y_0` is the initial parameter value, and :math:`r` is the common ratio.
    """

    def __init__(self, scheduler, r):
        super().__init__(scheduler)
        self.r = r

    def _update(self, param, t):
        updated_param = param * self.r
        return updated_param


if __name__ == "__main__":
    import inspect

    # Run this file to automatically generate the __all__ variable. Copy and past the output bellow.

    public_api = [
        name
        for name, obj in globals().items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__ == __name__
    ]
    print("__all__ = [")
    for name in sorted(public_api):
        print(f'    "{name}",')
    print("]")

__all__ = [
    "DeterministicTrend",
    "ExponentialDecay",
    "GeometricProgression",
    "IncrementUpdate",
    "NoUpdate",
    "OscillatingUpdate",
    "RandomWalk",
    "RandomWalkWithDrift",
    "RandomWalkWithDriftAndTrend",
    "StepWiseUpdate",
]
