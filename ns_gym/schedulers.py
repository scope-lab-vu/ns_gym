import ns_gym.base as base
import numpy as np

"""
The schdulers can be initialized at different times to we need to maintian an internal clock as well.
"""


class RandomScheduler(base.Scheduler):
    """Random event scheduler: Events occur randomly with a given probability at each time step.
    
    Args:
        probability (float): The probability of an event occurring at each time step.
        start (int, optional): The start time for the scheduler. Defaults to 0.
        end (int, optional): The end time for the scheduler. Defaults to infinity.
        seed (int, optional): Random generator seed. Defaults to None.

    """

    def __init__(
        self, probability: float = 0.5, start=0, end=np.inf, seed=None
    ) -> None:
        super().__init__(start, end)
        self.probability = probability
        self.rng = np.random.default_rng(seed=seed)

    def _check(self, t: int) -> bool:
        return self.rng.random() < self.probability


class CustomScheduler(base.Scheduler):
    """Custom event scheduler: Allows for custom event logic based on a user-defined function."""

    def __init__(self, event_function, start=0, end=np.inf) -> None:
        """
        Args:
            event_function (function): A function that takes the current time step as input and returns a boolean indicating whether an event should occur.
        """
        super().__init__(start, end)
        self.event_function = event_function

    def _check(self, t: int) -> bool:
        return self.event_function(t)


class ContinuousScheduler(base.Scheduler):
    """Continuous Event Scheduler : At every time step return true"""

    def __init__(self, start=0, end=np.inf) -> None:
        super().__init__(start, end)

    def _check(self, t: int) -> bool:
        return True


class DiscreteScheduler(base.Scheduler):
    """A discrete event scheduler returns a bool indicating where the system should transition at this time step

    Args:
        event_list (set): List of time steps to make a transition
    """

    def __init__(self, event_list: set, start=0, end=np.inf) -> None:
        super().__init__(start, end)
        self.event_list = event_list
        assert min(event_list) >= start, (
            "Scheduler start time occurs after first event in event list"
        )
        assert max(event_list) <= end, (
            "Scheduler end time occurs before last event in event list"
        )

    def _check(self, t: int) -> bool:
        return t in self.event_list


class PeriodicScheduler(base.Scheduler):
    """Periodic event scheduler: At periodic steps return true.

    Args:
        period (int): Period of event transition times.
    """

    def __init__(self, period: int, start=0, end=np.inf) -> None:
        super().__init__(start, end)
        self.period = period

    def _check(self, t: int) -> bool:
        return t % self.period == 0


class MemorylessScheduler(base.Scheduler):
    """Memoryless Scheduler: Events happen at intervals according to a Geometric distribution

    This scheduler models the number of trials that must be run before a success.
    The scheduler samples from a geometric distribution then records the new time an event will occur.
    After a transition we resample from the geometric distribution.

    Args:
        p (float): The probability of success of an individual trial.
        seed (int, optional): Random generator seed. Defaults to None.
    """

    def __init__(self, p: float, start=0, end=np.inf, seed=None) -> None:
        super().__init__(start, end)
        self.p = p
        self.rng = np.random.default_rng(seed=seed)
        self.transition_time = self.rng.geometric(p=self.p, size=(1,))

    def _check(self, t: int) -> bool:
        if t == self.transition_time:
            delta_t = self.rng.geometric(p=self.p, size=(1,))
            self.transition_time = delta_t + t
            return True
        else:
            return False


class BurstScheduler(base.Scheduler):
    """Burst event scheduler: fires for a window of consecutive steps, then stays silent, repeating cyclically.

    The cycle length is ``on_duration + off_duration``.  Within each cycle the
    scheduler fires for the first ``on_duration`` steps and is silent for the
    remaining ``off_duration`` steps.

    Args:
        on_duration (int): Number of consecutive steps to fire each cycle.
        off_duration (int): Number of consecutive silent steps each cycle.
        start (int, optional): Start time. Defaults to 0.
        end (int, optional): End time. Defaults to infinity.
    """

    def __init__(self, on_duration: int, off_duration: int, start=0, end=np.inf) -> None:
        super().__init__(start, end)
        self.on_duration = on_duration
        self.off_duration = off_duration
        self.cycle = on_duration + off_duration

    def _check(self, t: int) -> bool:
        return (t % self.cycle) < self.on_duration


class DecayingProbabilityScheduler(base.Scheduler):
    r"""Decaying probability scheduler: fires randomly with exponentially decaying probability.

    At each time step the probability of firing is:

    .. math::
        p(t) = p_0 \, e^{-\lambda t}

    where :math:`p_0` is the initial probability and :math:`\lambda` is the
    decay rate.  This models environments that stabilise over time.

    Args:
        initial_probability (float): Probability of firing at t = 0.
        decay_rate (float): Exponential decay rate (>= 0).
        start (int, optional): Start time. Defaults to 0.
        end (int, optional): End time. Defaults to infinity.
        seed (int, optional): Random generator seed. Defaults to None.
    """

    def __init__(
        self,
        initial_probability: float,
        decay_rate: float,
        start=0,
        end=np.inf,
        seed=None,
    ) -> None:
        super().__init__(start, end)
        self.initial_probability = initial_probability
        self.decay_rate = decay_rate
        self.rng = np.random.default_rng(seed=seed)

    def _check(self, t: int) -> bool:
        p = self.initial_probability * np.exp(-self.decay_rate * t)
        return bool(self.rng.random() < p)


class WindowScheduler(base.Scheduler):
    """Window-based event scheduler: fires only within specified time windows.

    Each window is an inclusive ``(start, end)`` tuple.  The scheduler fires
    at time ``t`` if ``t`` falls inside any window (and inside the global
    ``[start, end]`` range).

    Args:
        windows (list[tuple[int, int]]): List of ``(start, end)`` time windows.
        start (int, optional): Global start time. Defaults to 0.
        end (int, optional): Global end time. Defaults to infinity.
    """

    def __init__(self, windows: list, start=0, end=np.inf) -> None:
        super().__init__(start, end)
        self.windows = windows

    def _check(self, t: int) -> bool:
        return any(w_start <= t <= w_end for w_start, w_end in self.windows)


if __name__ == "__main__":
    print(base.Scheduler.__subclasses__())
