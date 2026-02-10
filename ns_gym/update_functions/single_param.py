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
        white_noise = self.rng.normal(self.mu, self.sigma)
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
        white_noise = self.rng.normal(self.mu, self.sigma)
        updated_param = param + white_noise
        return updated_param


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
        white_noise = self.rng.normal(self.mu, self.sigma)
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


class OrnsteinUhlenbeck(base.UpdateFn):
    r"""Mean-reverting update via the Ornstein-Uhlenbeck process.

    Overview:

        .. math::
            Y_t = Y_{t-1} + \theta (\mu - Y_{t-1}) + \sigma \epsilon_t

        where :math:`Y_t` is the parameter value at time step :math:`t`,
        :math:`\theta` controls the speed of mean reversion,
        :math:`\mu` is the long-run mean, and :math:`\sigma` scales the noise.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        theta (float): Speed of mean reversion.
        mu (float): Long-run equilibrium value.
        sigma (float): Volatility / noise scale.
        seed (Union[int, None], optional): Seed for the random number generator. Defaults to None.
    """

    def __init__(
        self,
        scheduler: Type[base.Scheduler],
        theta: float,
        mu: float,
        sigma: float = 0.0,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(scheduler)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed=seed)

    def _update(self, param: Any, t: int) -> Any:
        noise = self.rng.normal(0, self.sigma) if self.sigma > 0 else 0.0
        return param + self.theta * (self.mu - param) + noise


class SigmoidTransition(base.UpdateFn):
    r"""Smooth transition between two values via a logistic sigmoid.

    Overview:

        .. math::
            Y_t = a + \frac{b - a}{1 + \exp(-k (t - t_0))}

        The output replaces the current parameter value entirely (it is
        independent of :math:`Y_{t-1}`), producing a smooth S-curve from
        :math:`a` to :math:`b` centred at :math:`t_0`.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        a (float): Starting value (as :math:`t \to -\infty`).
        b (float): Ending value (as :math:`t \to +\infty`).
        k (float): Steepness of the transition. Larger = sharper.
        t0 (float): Midpoint time of the transition.
    """

    def __init__(
        self,
        scheduler: Type[base.Scheduler],
        a: float,
        b: float,
        k: float,
        t0: float,
    ) -> None:
        super().__init__(scheduler)
        self.a = a
        self.b = b
        self.k = k
        self.t0 = t0

    def _update(self, param: Any, t: Union[int, float]) -> Any:
        sigmoid = 1.0 / (1.0 + np.exp(-self.k * (t - self.t0)))
        return self.a + (self.b - self.a) * sigmoid


class CyclicUpdate(base.UpdateFn):
    r"""Cycle through a list of values, wrapping around when exhausted.

    Overview:
        Each time the update fires, the parameter is set to the next value
        in ``value_list``.  After reaching the end, the index wraps back to 0.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        value_list (list): Values to cycle through.
    """

    def __init__(self, scheduler: Type[base.Scheduler], value_list: list) -> None:
        super().__init__(scheduler)
        self.value_list = value_list
        self._index = 0

    def _update(self, param: Any, t: int) -> Any:
        val = self.value_list[self._index]
        self._index = (self._index + 1) % len(self.value_list)
        return val


class BoundedRandomWalk(base.UpdateFn):
    r"""Random walk clamped to ``[lo, hi]``.

    Overview:

        .. math::
            Y_t = \text{clip}(Y_{t-1} + \epsilon_t,\; lo,\; hi)

        where :math:`\epsilon_t \sim \mathcal{N}(\mu, \sigma^2)`.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        mu (float): Mean of the noise.
        sigma (float): Standard deviation of the noise.
        lo (float): Lower bound.
        hi (float): Upper bound.
        seed (Union[int, None], optional): Seed for the random number generator. Defaults to None.
    """

    def __init__(
        self,
        scheduler: Type[base.Scheduler],
        mu: float,
        sigma: float,
        lo: float,
        hi: float,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.lo = lo
        self.hi = hi
        self.rng = np.random.default_rng(seed=seed)

    def _update(self, param: Any, t: int) -> Any:
        noise = self.rng.normal(self.mu, self.sigma)
        return float(np.clip(param + noise, self.lo, self.hi))


class PolynomialTrend(base.UpdateFn):
    r"""Deterministic polynomial trend.

    Overview:

        .. math::
            Y_t = Y_{t-1} + \sum_{i=1}^{n} a_i \, t^{i}

        where ``coeffs = [a_1, a_2, ..., a_n]``.  A single-element list
        ``[slope]`` is equivalent to :class:`DeterministicTrend`.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        coeffs (list[float]): Polynomial coefficients ``[a_1, a_2, ...]`` for powers ``t, t^2, ...``.
    """

    def __init__(self, scheduler: Type[base.Scheduler], coeffs: list) -> None:
        super().__init__(scheduler)
        self.coeffs = coeffs

    def _update(self, param: Any, t: Union[int, float]) -> Any:
        trend = sum(a * t ** (i + 1) for i, a in enumerate(self.coeffs))
        return param + trend


class LinearInterpolation(base.UpdateFn):
    r"""Linearly interpolate from ``start_val`` to ``end_val`` over ``T`` steps.

    Overview:

        .. math::
            Y_t = start\_val + (end\_val - start\_val) \cdot \min\!\left(\frac{t}{T},\; 1\right)

        The output replaces the current parameter value entirely.
        After ``t >= T`` the value is clamped at ``end_val``.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        start_val (float): Value at ``t = 0``.
        end_val (float): Value at ``t = T``.
        T (int): Number of steps over which to interpolate.
    """

    def __init__(
        self,
        scheduler: Type[base.Scheduler],
        start_val: float,
        end_val: float,
        T: int,
    ) -> None:
        super().__init__(scheduler)
        self.start_val = start_val
        self.end_val = end_val
        self.T = T

    def _update(self, param: Any, t: Union[int, float]) -> Any:
        frac = min(t / self.T, 1.0)
        return self.start_val + (self.end_val - self.start_val) * frac


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
    "BoundedRandomWalk",
    "CyclicUpdate",
    "DecrementUpdate",
    "DeterministicTrend",
    "ExponentialDecay",
    "GeometricProgression",
    "IncrementUpdate",
    "LinearInterpolation",
    "NoUpdate",
    "OrnsteinUhlenbeck",
    "OscillatingUpdate",
    "PolynomialTrend",
    "RandomWalk",
    "RandomWalkWithDrift",
    "RandomWalkWithDriftAndTrend",
    "SigmoidTransition",
    "StepWiseUpdate",
]
