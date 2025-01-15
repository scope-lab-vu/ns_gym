import ns_gym.base as base
import ns_gym.utils as utils
from typing import Union, Any, Optional,Type
import numpy as np

'''
These are a collection of parameter update functions that can be used in the ns_bench framework.
These classes take in a single parameter (scalars) and return a new parameter. 
The update functions only "fire" when the scheduler returns True.

To maintain a consistent interface, the update functions can be implemented as a derived class of base.UpdateFn.

But really the only main requirement is that the update function takes in a parameter and a time step in the
__call__() method and returns a new parameter and a boolean value indicating whether the update function fired or not.


'''

# Clean this up and add more update functions.

##### Update Functions for single parameters (scalars) #####

class DeterministicTrend(base.UpdateFn):
    def __init__(self, scheduler: Type[base.Scheduler], slope: float) -> None:
        super().__init__(scheduler)
        self.slope = slope

    def __call__(self, param: float, t: float) -> tuple[float, bool]:
        return super().__call__(param, t)  

    def update(self, param: float, t: float) -> tuple[float, bool]:
        updated_param = param + self.slope * t
        return updated_param
    
class RandomWalkWithDriftAndTrend(base.UpdateFn):
    def __init__(self, scheduler: Type[base.Scheduler], alpha: float, mu: float, sigma: float, slope: float, seed: Union[int, None] = None) -> None:
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.slope = slope
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, param: float,t: float) -> Any:
        return super().__call__(param, t)
    
    def update(self, param: float, t: float) -> tuple[float, bool]:
            white_noise = self.rng.normal(self.mu, self.sigma, 1)
            updated_param = self.alpha + param + white_noise + self.slope * t
            return updated_param
    
class RandomWalk(base.UpdateFn):
    """Parameter update function that updates the parameter with white noise.    
    A pure random walk : Y_t = Y_{t-1} + \epsilon_t where Y_t is the parameter value at time step t
    and epsilon is white noise. 
    """
    def __init__(self,
                 scheduler: Type[base.Scheduler],
                 mu : Union[float,int] = 0,
                 sigma : Union[float,int] = 1,
                 seed = None) -> tuple[Any,bool]:
        """
        Args:
            mu (Union[float,int], optional): _description_. Defaults to 0.
            sigma (Union[float,int], optional): _description_. Defaults to 1.
            seed (_type_, optional): _description_. Defaults to None.

        Returns:
            tuple[Any,bool]: _description_
        """
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, 
                 param:Any,
                 t : Union[int,float]) -> tuple[float,bool]:
        return super().__call__(param,t)
    
    def update(self, param: Any, t: Union[int,float]) -> Any:
        white_noise = self.rng.normal(self.mu, self.sigma, 1)
        updated_param = param + white_noise
        return updated_param[0]

class RandomWalkWithDrift(base.UpdateFn):
    def __init__(self,
                 scheduler: Type[base.Scheduler],
                 alpha: float,
                 mu: float,
                 sigma: float,
                 seed: Union[int,None] = None) -> None:
        """_summary_

        Args:
            alpha (float): _description_
            mu (float): _description_
            sigma (float): _description_
            seed (int): _description_
        """
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self,
                 param:Any,
                 t : float) -> Any:
        return super().__call__(param,t)
    
    def update(self, param: Any, t: int) -> Any:
        white_noise = self.rng.normal(self.mu,self.sigma,1)
        upated_param = self.alpha + param + white_noise
        return upated_param
    
class IncrementUpdate(base.UpdateFn):
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

    def __call__(self, param: Any, t: int) -> Any:
        return super().__call__(param, t)   
        
    def update(self, param: Any, t: int) -> Any:
        # check if action is in kwargs and if so set the intended direction to the action
        param+=self.k
        return param
    
class DecrementUpdate(base.UpdateFn):
    """Decrment the probabilty go going in the intened direction by some k.

    #TODO: instead of hard coding the intended direction, we can pass in the action as a param
        This action will be used to determine the intended direction. This works in our discrete action case. 

    Note:
        This function only really works in the Frozen lake environment where the intended directino is always in the second elem in the prob array.
    """
    def __init__(self,scheduler,k) -> None:
        super().__init__(scheduler)
        self.k = k

    def __call__(self,param,t) -> Any:
        return super().__call__(param,t)
        
    def update(self,param,t) -> Any:
        param-=self.k
        return param
    
class StepWiseUpdate(base.UpdateFn):
    """Update the parameter at specific time steps.
    """
    def __init__(self, scheduler: Type[base.Scheduler], param_list: list) -> None:
        super().__init__(scheduler)
        self.param_list = param_list

    def __call__(self, param: Any, t: int) -> Any:
        return super().__call__(param, t)   
        
    def update(self, param: Any, t: int) -> Any:
        try:
            param = self.param_list.pop(0)
        except AssertionError:
            "No more parameters to update"
        finally:
            return param
        

class NoUpdate(base.UpdateFn):
    """Do not update the parameter but return correct interface
    """

    def __init__(self, scheduler: Type[base.Scheduler]) -> None:
        super().__init__(scheduler)
    
    def __call__(self, param: Any, t: int) -> Any:
        return super().__call__(param, t)
    
    def update(self, param: Any, t: int) -> Any:
        return param
    

class OscillatingUpdate(base.UpdateFn):
    """Update the parameter with an oscillating function.
    """
    def __init__(self, scheduler: Type[base.Scheduler], delta: float) -> None:
        super().__init__(scheduler)
        self.delta = delta

    def __call__(self, param: Any, t: int) -> Any:
        return super().__call__(param, t)
    
    def update(self, param: Any, t: int) -> Any:
        oscillation = self.delta * np.sin(t)
        return param + oscillation


if __name__ == "__main__":
    from ns_gym.schedulers import PeriodicScheduler,DiscreteScheduler
    from ns_gym.update_functions.single_param import StepWiseUpdate
    import ns_gym.wrappers as wrappers
    import gymnasium as gym
    import itertools
    
    # scheduler1 = PeriodicScheduler(period = 2)
    scheduler1 = DiscreteScheduler({0,5})
    update_fn = StepWiseUpdate(scheduler1,[1,0.8])
    
    env = gym.make("FrozenLake-v1", render_mode = "human")
    env = wrappers.NSFrozenLakeWrapper(env,update_fn) 

   

    while True:
        obs,info = env.reset()
        print(obs)
        print(info)
        policy = [1,1,2,1,2,2]
        for a in itertools.cycle(policy):
            obs,reward,terminated,truncated,info = env.step(a)
            print(f"t:{obs.relative_time},info:{info}")
            if terminated:
                break

        


# daw