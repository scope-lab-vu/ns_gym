import numpy as np
from abc import ABC
from typing import Optional,Callable,Union,Any,Type,SupportsFloat
from dataclasses import dataclass
from gymnasium import Env, Wrapper
import gymnasium as gym
import numpy as np
import ns_gym.utils as utils
import uuid 
import copy
from scipy.stats import wasserstein_distance

#TODO: add other envs
TUNABLE_PARAMS = {"CartPoleEnv": {"gravity":9.8,"masscart":1.0,"masspole":0.1,"force_mag": 10.0,"tau":0.02,"length":0.5},
                  "AcrobotEnv": {"dt":0.2,"LINK_LENGTH_1":1.0,"LINK_LENGTH_2":1.0,"LINK_MASS_1":1.0,"LINK_MASS_2":1.1,"LINK_COM_POS_1":0.5,"LINK_COM_POS_2":0.5,"LINK_MOI":1.0},
                  "MountianCarEnv": {"gravity":0.0025,"force":0.001},
                  "PendulumEnv": {"m":1.0,"l":1.0,"dt":0.05,"g":9.8},
                  }
"""
Look up table for tunable parameters and their default values for each environment.
"""


"""
Some core types 
"""

@dataclass(frozen=True)
class Observation:
    """Observation dataclass type. This is the output of the step function in the environment.

    Attributes:
        state (Union[np.ndarray,int]): The state of the environment
        env_change (Union[dict[str, bool],None]): A dictionary of boolean flags indicating what param of the environment has changed.
        delta_change (Union[float]): The amount of change in the transition function of the environment
        relative_time (Union[int,float]): The relative time of the observation since the start of the environment episode.
    """
    state : Union[np.ndarray,int]
    env_change: Union[dict[str, bool],None] 
    delta_change: Union[dict[float,float],float,None]
    relative_time: Union[int,float]


@dataclass(frozen=True)
class Reward:
    """Reward dataclass type. This is the output of the step function in the environment.

    Attributes:
        reward (Union[int,float]): The reward of the environment
        env_change (Union[dict[str, bool],None]): A dictionary of boolean flags indicating what param of the environment has changed.
        delta_change (Union[float]): The change in the reward function of the environment.
        relative_time (Union[int,float]): The relative time of the observation since the start of the environment episode.
    """
    reward: Union[int,float]
    env_change: dict[str, bool]
    delta_change: Union[float,None]
    relative_time: Union[int,float]

class Scheduler(ABC):
    """Base class for scheduler functions. This class is used to determine when to update a parameter in the environment.
    """
    def __init__(self,
                 start = 0, 
                 end = np.inf) -> None:
        super().__init__()
        self.start =  start
        self.end = end

    def __call__(self, 
                 t:int) -> bool:
        """
        Args:
            t (int): MDP timestep

        Returns:
            bool: Boolean flag indicating whether to update the parameter or not.
        """
        # return super().__call__()
        return NotImplementedError("Subclasses must implement this method")


class UpdateFn(ABC):
    """Base class for update functions that update a single parameter. Updates a scalar parameter
    """
    def __init__(self, scheduler: Type[Scheduler]) -> None:
        """
        Args:
            scheduler (Type[Scheduler]): scheduler object that determines when to update the parameter

        Attributes:
            prev_param: The previous parameter value
            prev_time: The previous time the parameter was updated
        """

        assert isinstance(scheduler, Scheduler), (f"Expected scheduler to be a subclass of Scheduler, got {type(scheduler)}")
        self.scheduler = scheduler
        self.prev_param = None
        self.prev_time = -1

    def __call__(self,param:Any,t:Union[int,float]) -> tuple[Any,bool,float]:
        """Update the parameter if the scheduler returns True

        Args:
            param (Any): The parameter to be updated
            t (Union[int,float]): The current time step
        
        Returns:
            Any: The updated parameter
            bool: Boolean flag indicating whether the parameter was updated or not
            float: The amount of change in the parameter
        """
        assert isinstance(t,(int,float)),(f"Expected t to be an int or float, got {type(t)}, Arrays operations need to inherit from UpdateDistributionFn")
        if self.scheduler(t):
            updated_param = self.update(copy.copy(param),t)   

            delta_change = self._get_delta_change(param,updated_param,t)
            self.prev_param = param
            self.prev_time = t
            return updated_param, True, delta_change
        else:       
            self.prev_param = param
            self.prev_time = t  
            return param, False, None
        
    def update(self, param:Any, t:int) -> Any:
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_delta_change(self, param:Any,updated_param:Any, t:int) -> float:

        return updated_param - param

    
class UpdateDistributionFn(UpdateFn):
    """Base class for all update functions that update a distribution represented as a list
    """
    def __call__(self,param:Any,t:Union[int,float]) -> Any:
        assert(isinstance(param,list)),(f"param must be a list, got {type(param)}")
        return super().__call__(param,t)
    
    def _get_delta_change(self, param: Any,updated_param:Any, t: int) -> float:
        """We will use the Wasserstein distance to measure the amount of change in the distribution.

        Args:
            param (Any): The parameter to be updated
            t (int): The current time step

        Returns:
            float: Amount of change in the distribution

        """
        return utils.wasserstein_distance(param,updated_param)


class NSWrapper(Wrapper):
    """Base class for non-stationary wrappers
    """
    def __init__(self, 
                 env: Type[Env], 
                 tunable_params: dict[str,Union[Type[UpdateFn], Type[UpdateDistributionFn]]], 
                 change_notification: bool = False, 
                 delta_change_notification: bool = False, 
                 in_sim_change: bool = False, 
                 **kwargs: Any):
        """
        Args:
            env (Env): Gym environment
            tunable_params (dict[str,Union[Type[UpdateFn],Type[UpdateDistributionFn]]): Dictionary of parameter names and their associated update functions.
            change_notification (bool): Sets a basic notification level. Returns a boolean flag to indicate whether to notify the agent of changes in the environment. Defaults to False.
            delta_change_notification (bool): Sets detailed notification levle. Returns Flag to indicate whether to notify the agent of changes in the transition function. Defaults to False.
            in_sim_change (bool): Flag to indicate whether to allow changes in the environment during simulation (e.g MCTS rollouts). Defaults to False.

        Attributes:
            frozen (bool): Flag to indicate whether the environment is frozen or not.
            is_sim_env (bool): Flag to indicate whether the environment is a simulation environment or not.

        """
        Wrapper.__init__(self,env)
        #was super().__init__(env) before
        if delta_change_notification: assert(change_notification), "If change_notification is True, delta_change_notification must be True"
        self.tunable_params = tunable_params
        self.init_initial_params = copy.deepcopy(self.tunable_params)
        self.change_notification = change_notification
        self.delta_change_notification = delta_change_notification
        self.in_sim_change = in_sim_change
        self.frozen = False
        self.is_sim_env = False
        self.t = 0
        self.has_reset = False
    
    def step(self, action: Any, env_change: dict[str,bool], delta_change: dict[str,bool]) -> tuple[Type[Observation], Type[Reward], bool, bool, dict[str, Any]]:
        """Step function for the environment. Augments observations and rewards with additional information about changes in the environment and transition function.

        This function is called by to take a step in the environment. We augment the standard gym observation and reward with additional information about changes in the environment and transition function.
        Additionally we increment the relative time of the environment. Also if the change_notification flag is set to False, we do not notify the agent of changes in the environment. 
        Similarly if the delta_change_notification flag is set to False, we do not notify the agent of the amount of change in the transition function.

        Args:
            action (int): Action taken by the agent
            env_chage (dict[str,bool]): Envrioment change flags. Keys are parameter names and values are boolean flags indicating whether the parameter has changed.
            delta_change (dict[str,bool]): The amount of change a parameter has undergone. Keys are parameter names and values are the amount of change.

        Returns:
            tuple[Type[Observation], Type[Reward], bool, bool, dict[str, Any]]: Observation, reward, termination flag, truncation flag, and additional information.

        """

        state,reward,terminated,truncated,info = super().step(action)
        self.t+=self.delta_t

        if not self.change_notification:
            env_change = None
        
        if not self.delta_change_notification:
            delta_change = None

        obs  = Observation(state = state,
                           env_change = env_change,
                           delta_change = delta_change,
                           relative_time = self.t)
        
        rew = Reward(reward = reward,
                        env_change = env_change,
                        delta_change = delta_change,
                        relative_time = self.t)
        
        return obs, rew, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        state, info = super().reset(seed=seed, options=options)
        self.has_reset = True
        self.t = 0
        self.tunable_params = copy.deepcopy(self.init_initial_params)
        return Observation(state=state,env_change=None,delta_change=None,relative_time=0), info
    
    def freeze(self, mode: bool = True):
        """Freeze the environment dynamics for simulation.

        This function "freezes" the current MDP so that the environment dynamics do not change. This is necessary for decion making when the agent is unaware of future changes in the environment.
        Essentially we can use this function to take a snapshot of the current MDP and use it for decision making.
        Inspired by the pytorch nn.Module.train() and nn.Module.eval() functions.
        """
        if not isinstance(mode,bool):
            raise TypeError(f"Expected mode to be a boolean, got {type(mode)}")
        self.forzen = mode
        return self
    
    def unfreeze(self):
        """Unfreeze the environment dynamics for simulation.

        This function "unfreezes" the current MDP so that the environment dynamics can change. 
        """
        return self.freeze(False)
    
    def __deepcopy__(self,memo):
        """Keeps track of deepcopying for the environment. 
        
        If a derived class of this environement is made we set a flag to indicate that the environment is the simulation environment.

        This is the intended behavior for the deepcopy function.
        ```python
        env = gym.make("FrozenLake-v1")
        env = NSFrozenLakeWrapper(env,updatefn,is_slippery=False)
        sim_env = deepcopy(env)
        ```
        Then `sim_env.is_sim_env` will be set to True.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_planning_env(self):
        """Get the planning environment. 

        This function returns a copy of the current environment in its current state but the "transition function" is set to the initial transition function.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_default_params(self):
        """Get dictionary of default parameters and their initial values
        """
        try:
            return TUNABLE_PARAMS[self.unwrapped.__class__.__name__]
        except KeyError:
            raise NotImplementedError(f"Default parameters for {self.unwrapped.__class__.__name__} not included in TUNABLE_PARAMS. Please add them to the dictionary in the classic_control.py file.")

    def __repr__(self):
        return super().__repr__()
    
    def __str__(self):
        """Change the string representation of the environment so that user can see what/how parameters are being updated.
        """
        # TODO: Implement this function
        return super().__str__()
    
if __name__ == "__main__":
    pass