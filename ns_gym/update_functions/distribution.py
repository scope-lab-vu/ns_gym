from rich import inspect
import ns_gym.base as base
import ns_gym.utils as utils
from typing import Any, Type, Union, Optional
import numpy as np

"""
These classes update the probability distributions represented as a lists.
"""


class RandomCategorical(base.UpdateDistributionFn):
    """Update the distirbution as a random categorical distribution.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        seed (Optional[int], optional): Seed for the random number generator. Defaults to None.

    Note:
        This update function would return a new random categorical distribution.
        The new categorical distribution is sampled from a Dirichlet distribution with all parameters equal to 1.

    """

    def __init__(self, scheduler: base.Scheduler, seed: Optional[int] = None) -> None:
        super().__init__(scheduler)
        self.rng = np.random.default_rng(seed=seed)

    def _update(self, param, t: int) -> Any:
        """Update the parameter by returning a new uniform random categorical distribution.

        Args:
            param (list): parameter to be updated
            t (int): current time step

        Returns:
            Any: updated parameter
        """
        return list(self.rng.dirichlet(np.ones(len(param))))


class DistributionIncrementUpdate(base.UpdateDistributionFn):
    """Increment the the parameter by k.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        k (float): The amount which the parameter is updated.

    Note:
        This update function is useful for testing the robustness of the agent to changes in the environment.
        If the parameter is a probability, k would update the probability of going in the intended direction.
        Otherwise, k would be added to the parameter's value.
    """

    def __init__(self, scheduler: Type[base.Scheduler], k: float) -> None:
        super().__init__(scheduler)
        self.k = k

    def __call__(self, param: list[float], t: int) -> Any:
        return super().__call__(param, t)

    def _update(self, param: list[float], t: int, **kwargs) -> Any:
        """Update the parameter by incrementing the intended direction by k.
        """
        param[0] = min(1, param[0] + self.k)
        for i in range(1, len(param)):
            param[i] = (1 - param[0]) / (len(param) - 1)
        return param


class DistributionDecrementUpdate(base.UpdateDistributionFn):
    """Decrement the probability of going in the intended direction by some k.

    Overview:
        This function is used to decrement the probability distribution by some k. The probability distribution is represented as a list of probabilities. The intended direction is the first element in the probability distribution.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        k (float): The amount which the parameter is updated.
    """

    def __init__(self, scheduler: base.Scheduler, k: float) -> None:
        super().__init__(scheduler)
        self.k = k

    def __call__(self, param: list[float], t: int) -> Any:
        return super().__call__(param, t)

    def _update(self, param: list[float], t: int, **kwargs) -> Any:
        """Update the parameter by decrementing the intended direction by k.

        Returns:
            param (list): Updated probability distribution
        """
        param[0] = max(0, param[0] - self.k)
        for i in range(1, len(param)):
            param[i] = (1 - param[0]) / (len(param) - 1)
        return param


class DistributionStepWiseUpdate(base.UpdateDistributionFn):
    """Update the parameter to values to a set of predefined values at specific time steps.

    Args:
        scheduler (base.Scheduler): scheduler that determines when the update function fires.
        update_values (list): A list of values that the parameter is updated to at specific time steps.
    
    """

    def __init__(self, scheduler: base.Scheduler, update_values: list) -> None:
        super().__init__(scheduler)
        self.update_values = update_values

    def __call__(self, param: Any, t: int) -> Any:
        return super().__call__(param, t)

    def _update(self, param, t: int) -> Any:
        """
        Args:
            param (list): current parameter value
            t (int): current time step

        Returns:
            list: updated parameter value
        """
        try:
            param = self.update_values.pop(0)
        except AssertionError:
            "No more parameters to update"
        finally:
            return param


class LCBoundedDistrubutionUpdate(base.UpdateDistributionFn):
    """Decrement the parameters so that the change is Lipshitz continuous.

    Overview:
        This function would call the decrement update function and check if the change is Lipshitz continuous.
        If not it would recall the decrement update function until the change is Lipshitz continuous.

        The Lipshitz continuous constraint between to probability distributions is defined as:

         .. math::
            W_1(p_t(.|s,a),p_{t'}(.|s,a)) <= L * |t - t'|


        Where :math:`W_1` is the Wasserstein distance between two probability distributions.

    Args:
        update_fn (Type[base.UpdateDistributionFn]): The update function that updates the parameter.
        L (float): The Lipshitz constant.

    Note:
        This update function is an implementation of transition fucntion in Lecarpentier and Rechelson et al. 2019
    """

    def __init__(self, scheduler, L: float, update_fn=None) -> None:
        super().__init__(scheduler)
        self.L = L
        if update_fn is None:
            self.update_fn = RandomCategorical(scheduler)
        else:
            assert issubclass(update_fn, base.UpdateDistributionFn), (
                "update_fn must be a subclass of base.UpdateDistributionFn"
            )
            self.update_fn = update_fn(scheduler)

    def _update(self, param: Any, t: int) -> Any:
        max_trys = 1e5
        count = 0
        cur_dist = param
        updated_dist = self.update_fn.update(param, t)
        wass_dist = utils.wasserstein_distance(cur_dist, updated_dist)

        delta_time = abs(t - self.prev_time)
        d = self.L * delta_time
        while wass_dist > d and count < max_trys:
            updated_dist = self.update_fn.update(param, t)
            wass_dist = utils.wasserstein_distance(cur_dist, updated_dist)
            count += 1

        if count >= max_trys:
            raise ValueError("Could not find a Lipshitz continuous update")
        else:
            return updated_dist


class BudgetBoundedIncrement(base.UpdateDistributionFn):
    """Increment the parameters so that the total amount of change is bounded by some budget.

    Overview:
        This function contrains the total amount of change in the parameter by some max budget.
        This formulation is outlined in Cheung et al. 2020.

    Args:
        scheduler (base.Scheduler): scheduler that determines when the update function fires.
        k (float): The amount which the parameter is updated.
        B (Union[int,float]): The maximum total amount of change allowed in the parameter.
    """

    def __init__(
        self, scheduler: base.Scheduler, k: float, B: Union[int, float]
    ) -> None:
        super().__init__(scheduler, k)
        self.B = B
        self.total_change = 0

    def __call__(self, param: Any, t: int) -> Any:
        curr_dist = param
        updated_param, change = super().__call__(param, t)
        amount_change = utils.wasserstein_distance(curr_dist, updated_param)
        if self.total_change + amount_change <= self.B:
            self.total_change += amount_change
            return updated_param, change
        else:
            return curr_dist, False


class DistributionNoUpdate(base.UpdateDistributionFn):
    """Does not update the parameter but return correct ns_bench interface.

    Overview:
        This function does not update the parameter.
    """

    def __init__(self, scheduler: base.Scheduler) -> None:
        super().__init__(scheduler)

    def __call__(self, param: Any, t: int) -> Any:
        return super().__call__(param, t)

    def _update(self, param: Any, t: int) -> Any:
        return param


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
    "BudgetBoundedIncrement",
    "DistributionDecrementUpdate",
    "DistributionIncrementUpdate",
    "DistributionNoUpdate",
    "DistributionStepWiseUpdate",
    "LCBoundedDistrubutionUpdate",
    "RandomCategorical",
]
