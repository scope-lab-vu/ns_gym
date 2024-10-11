import ns_gym.base as base
import ns_gym.utils as utils
from typing import Any, Type, Union, Optional
import numpy as np

"""
These classes update the probability distributions represented as a lists.

Each class is a subclass of the base.UpdateDistributionFn class. 

Each class has a scheduler that determines when the update function fires.
Additionally, each class has a method update that updates the parameter based on the scheduler.
The param is list of probabilities. Where the first elemetn of param is the probability of taking the intended action. 
The second elemeent is the probablity of taking the action encodes as (action - 1) % self.action_space.n. 
That means in a gridworld setting (FrozenLake, Cliffwalkingm, bridge) the intend direction is the first element. 
The second element is the probabilty of going in a positive 90 degreee direction from the intended direction.
The third element is the probability of going in a 180 degree direction from the intended direction.
The fourth element is the probability of going in a negative 90 degree direction from the intended direction.
"""    

class RandomCategroical(base.UpdateDistributionFn):
    """Update the distirbution as a random categorical distribution.
    """

    def __init__(self, scheduler: base.Scheduler,seed: Optional[int] = None) -> None:
        super().__init__(scheduler)
        self.rng = np.random.default_rng(seed=seed)
    
    def update(self, param: Any, t: int) -> Any:
        """Update the parameter by returning a new uniform random categorical distribution.

        Args:
            param (Any): _description_
            t (int): _description_

        Returns:
            Any: _description_
        """
        return list(self.rng.dirichlet(np.ones(len(param))))
        




class DistributionIncrementUpdate(base.UpdateDistributionFn):
    """Increment the the parameter by k.

    Note:
        If the parameter is a probability, k would update the probability of going in the intended direction.
        Otherwise, k would be added to the parameter.
    """
    def __init__(self, 
                 scheduler: Type[base.Scheduler],
                 k: float) -> None:
        """
        Args:
            scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
            k (float): The amount which the parameter is updated.
        """
        super().__init__(scheduler)
        self.k = k

    def __call__(self, param: list[float], t: int) -> Any:
        return super().__call__(param, t)   
        
    def update(self, param: list[float], t: int, **kwargs) -> Any:
        """Update the parameter by incrementing the intended direction by k.

        Args:
            param (Any): _description_
            t (int): _description_

        Returns:
            param (list): Updated probability distribution
        """
        param[0] = min(1,param[0] + self.k)
        for i in range(1,len(param)):
            param[i] = (1-param[0])/(len(param)-1)
        return param

class DistributionDecrementUpdate(base.UpdateDistributionFn):
    """Decrement the probability of going in the intended direction by some k.

    Overview:
        This function is used to decrement the probability distribution by some k. 
        The probability distribution is represented as a list of probabilities. 
        The intended direction is the first element in the probability distribution. 
    """
    def __init__(self, scheduler: base.Scheduler, k: float) -> None:
        super().__init__(scheduler)
        self.k = k

    def __call__(self, param: list[float], t: int) -> Any:
        return super().__call__(param, t)
    
    def update(self, param: list[float], t: int, **kwargs) -> Any:
        """Update the parameter by decrementing the intended direction by k.

        Args:
            param (Any): _description_
            t (int): _description_

        Returns:
            param (list): Updated probability distribution
        """
        param[0] = max(0,param[0] - self.k)
        for i in range(1,len(param)):
            param[i] = (1-param[0])/(len(param)-1)
        return param

class DistributionStepWiseUpdate(base.UpdateDistributionFn):
    """Update the parameter to values to a set of predefined values at specific time steps.
    TODO: Implement this
    """
    def __init__(self, scheduler: base.Scheduler,update_values:list) -> None:
        """
        Args:
            scheduler (base.Scheduler): scheduler that determines when the update function fires.
            update_values (list): A list of values that the parameter is updated to at specific time steps. 
                update_values[0] is the first value updated update_values[-1] is the last value updated.
        """
        super().__init__(scheduler)
        self.update_values = update_values

    def __call__(self,param:Any, t:int) -> Any:
        return super().__call__(param,t)
    
    def update(self, param: Any, t: int) -> Any:
        """TODO implement this

        Args:
            param (Any): _description_
            t (int): _description_

        Returns:
            Any: _description_
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
            W_1(p_t(.|s,a),p_{t'}(.|s,a)) <= L * |t - t'|

        Where W_1 is the Wasserstein distance between two probability distributions.

    Note:
        This update function is an implementation of transition fucntion in Lecarpentier and Rechelson et al. 2019
    """
    def __init__(self,scheduler,L:float, update_fn=None) -> None:
        """
        Args:
            update_fn (Type[base.UpdateDistributionFn]): The update function that updates the parameter.
            L (float): The Lipshitz constant.
        """
        super().__init__(scheduler)
        self.L = L  
        if update_fn is None:
            self.update_fn = RandomCategroical(scheduler)
        else:
            assert issubclass(update_fn, base.UpdateDistributionFn), "update_fn must be a subclass of base.UpdateDistributionFn"
            self.update_fn = update_fn(scheduler)

    def __call__(self, param, t) -> Any:
        """Update the parameter so that the change is Lipshitz continuous.
        
        Args:
            param (_type_): _description_
            t (_type_): _description_

        Returns:
            Any: _description_

        Note:
        """
        return super().__call__(param, t)
    
    def update(self, param: Any, t: int) -> Any:

        max_trys = 1e5
        count = 0
        cur_dist = param
        updated_dist = self.update_fn.update(param, t)
        wass_dist = utils.wasserstein_distance(cur_dist, updated_dist)

        delta_time = abs(t-self.prev_time)
        d = self.L  * delta_time
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
    """
    def __init__(self, scheduler: base.Scheduler, k: float, B: Union[int,float]) -> None:
        """

        Args:
            scheduler (base.Scheduler): scheduler that determines when the update function fires.
            k (float): The amount which the parameter is updated.
            B (Union[int,float]): The maximum total amount of change allowed in the parameter.
        """
        super().__init__(scheduler, k)
        self.B = B
        self.total_change = 0
    
    def __call__(self,param: Any,t: int) -> Any:
        curr_dist = param 
        updated_param, change =  super().__call__(param,t)
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
    
    def update(self, param: Any, t: int) -> Any:
        return param
    

