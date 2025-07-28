from typing import Any, Type, Union
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from gymnasium import Wrapper
from copy import deepcopy
import warnings

from ns_gym.utils import update_probability_table,state_action_update,n_choose_k
import ns_gym.utils as utils 
import ns_gym.base as base

"""
Table of tunable parameters for the classic control suite of environments
Key : Class name of the environment
Value : Dictionary of tunable parameters and their initial values
"""
class ConstraintViolationWarning(Warning):
    """Warning issued when a constraint in the application is violated."""
    pass
"""
Wrapper for the classic control suite of environments.
"""



class NSClassicControlWrapper(base.NSWrapper):
    """ 
    Overview:
        This is a proof of concept implementation of a non-statinoarity wrapper for the classic control suite of envs.
        The wrapper creates non-stationary transition functions by varying input parameter of the dynamical equation with respect to time.
        The way thsee parameters evolve w.r.t time is defined by a user specified transition function. 

        This wrapper modifies two methods of the standard gym environments. 
             1. step() 
             2. reset()
        
        In addition to altering the physics of each episode step this wrapper augments the observations with, time and 
        a boolean flag denoting there has been a change in enviroment dynamics.
        ```
    """
    def __init__(self, 
                 env: Type[gym.Env], 
                 tunable_params: dict[str,Union[Type[base.UpdateFn], Type[base.UpdateDistributionFn]]], 
                 change_notification: bool = False, 
                 delta_change_notification: bool = False, 
                 in_sim_change: bool = False, 
                 **kwargs: Any):
        """
        Args:
            env (gym.Env): Base gym environment.
            tunable_params (dict[str,Type[base.NSTransitionFn]]): Dictionary of parmaeter names and their associated update functions. 

        Keyword Args:
            change_notification (bool): Flag to indicate whether to notify the agent of changes in the environment. Defaults to False.
            delta_change_notification (bool): Flag to indicate whether to notify the agent of changes in the transition function. Defaults to False.
        """

        assert(env.unwrapped.__class__.__name__ in base.TUNABLE_PARAMS.keys()), f"{env.unwrapped.__class__.__name__} is not a supported environment"
        super().__init__(env = env,
                         tunable_params = tunable_params,
                         change_notification = change_notification, 
                         delta_change_notification = delta_change_notification,
                         in_sim_change = in_sim_change,
                         **kwargs)
        self.t = 0
        self.delta_t = 1
        
        self.initial_params = {}
        for key in tunable_params.keys():
            assert(key in base.TUNABLE_PARAMS[self.unwrapped.__class__.__name__].keys()), f"{key} is not a tunable parameter for {self.unwrapped.__class__.__name__}"

            self.initial_params[key] = deepcopy(getattr(self.unwrapped,key))

    def step(self, 
             action: Union[float,int]) -> tuple[base.Observation, base.Reward, bool, bool, dict[str, Any]]:
        """Step through environment and update environmental parameters

        Args:
            action (Union[float,int]): Action to take in environment

        Returns:
            tuple[base.Observation, base.Reward, bool, bool, dict[str, Any]]: NS-Gym Observation type, reward, done flag, truncated flag, info dictionary
        """
        if self.is_sim_env and not self.in_sim_change:
            obs,reward,terminated,truncated,info = super().step(action,env_change=None,delta_change=None)
        else:                
            env_change = {}
            delta_change = {}
            new_vals = {} 
            for p,fn in self.tunable_params.items():
                cur_val = getattr(self.unwrapped,p)
                new_val, change_flag, delta = fn(cur_val,self.t) 
                delta_change[p] = delta
                env_change[p] = change_flag
                new_vals[p] = new_val
            
            for k,v in self._constraint_checker(new_vals).items():
                if not v: # If the constraint is not violated, update the parameter
                    setattr(self.unwrapped,k,new_vals[k])
                else:
                    delta_change[k] = False
                    env_change[k] = False
        
            self._dependency_resolver()
            obs,reward,terminated,truncated,info = super().step(action,env_change=env_change,delta_change=delta_change)
            info["prob"] = 1.0
    
        return obs,reward,terminated,truncated,info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[base.Observation, dict[str, Any]]:
        """Reset environment
        """
        obs,info = super().reset(seed=seed, options=options)
        for k,v in self.initial_params.items():
            setattr(self.unwrapped,k,deepcopy(v))   

        self.t = 0 
        return obs,info 
    
    def close(self):
        return super().close()
    
    def __str__(self):
        return super().__str__()
    
    def __repr__(self):
        return super().__repr__()
    
    def get_planning_env(self):
        """Return a copy of the environment
        NOTE: 
            - If the environment is a simulation environment, the function returns a deepcopy of the simulation environment.
            - If change notification is enabled, the function returns a deepcopy of the current environment because the decision making agent needs to be aware of the changes in the environment.
            - If change notification is disabled, the function returns a deepcopy of the environment with the initial parameters.
        """
        assert(self.has_reset),("The environment must be reset before getting the planning environment.")
        if self.is_sim_env or self.change_notification: 
            return deepcopy(self)
        elif not self.change_notification:
            planning_env = deepcopy(self)
            for k,v in self.initial_params.items():
                setattr(planning_env.unwrapped,k,deepcopy(v))
            return planning_env
    
    def __deepcopy__(self, memo):
        sim_env = gym.make(self.unwrapped.spec.id)
        sim_env = NSClassicControlWrapper(sim_env,
                                          deepcopy(self.tunable_params),
                                          self.change_notification,
                                          self.delta_change_notification,
                                          self.in_sim_change)
        sim_env.reset()
        sim_env.unwrapped.state = deepcopy(self.unwrapped.state)
        sim_env.t = deepcopy(self.t)
        for k,v in self.tunable_params.items():
            setattr(sim_env.unwrapped,k,deepcopy(getattr(self.unwrapped,k)))
        sim_env._dependency_resolver()
        sim_env.is_sim_env = True
        return sim_env        

    def get_default_params(self):
        """Get dictionary of default parameters and their initial values
        """
        return super().get_default_params()
    
    def _constraint_checker(self,new_vals) -> dict[str,bool]:
        """Check if the physical constraints of the environment are being violated, and all dependent parameters are updated accordingly. 
        Checks evironment parameters after each update step. If a constraint is violated, the parameter does not update and a warning is issued.

        Args:
            new_vals (dict[str,float]): New value of the parameter.

        Returns:
            constraint_dict (dict[str,bool]): Dictionary of parameters and their constraint violation status. True is a constraint is violated, False otherwise.
        Note: 
            Since each environement has different physical contraints, I can either create a new class for of each environment or just implement this method in the wrapper 
            that check the base environment name.

        - Relook at contrains to see if they make sense, no division by zero, no negative values, etc.
        - Make sure all dependent parameters are updated accordingly.
        - Should we store the previous values of the parameters?
        """
        constraint_dict: dict[str,bool] = {}

        if self.unwrapped.__class__.__name__ == "CartPoleEnv":
            for p,v in new_vals.items():
                constraint_dict[p] = False
                if p == "length"  and v <= 0:
                    warnings.warn("Length of the pole cannot be negative, length not updated.",ConstraintViolationWarning)
                    constraint_dict[p] = True
                elif p == "masscart" and v <= 0:
                    warnings.warn("Mass of the cart must be greater than zero, cart mass not updated",ConstraintViolationWarning)
                    constraint_dict[p] = True
                elif p == "masspole" and v <= 0:
                    warnings.warn("Mass of the pole must be greater than zero, pole mass not updated",ConstraintViolationWarning)
                    constraint_dict[p] = True
                elif p == "gravity"  and v < 0:
                    warnings.warn("Gravity cannot be negative, gravity not updated",ConstraintViolationWarning)
                    constraint_dict[p] = True
            return constraint_dict

        elif self.unwrapped.__class__.__name__ == "AcrobotEnv": 
            for p,new_val in new_vals.items():
                constraint_dict[p] = False

                if p == "LINK_LENGTH_1":
                    if new_val <= 0: # Make sure the length of the link is not negative
                        warnings.warn("Length of link must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                    elif "LINK_COM_POS_1" in new_vals.keys() and new_vals["LINK_COM_POS_1"] > new_val:
                        warnings.warn("Length of link must be greater than the position of its center of mass, link 1 length parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                    elif new_val < self.unwrapped.LINK_COM_POS_1:
                        warnings.warn("Length of link must be greater than the position of its center of mass, link 1 length parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue

                elif p == "LINK_LENGTH_2" and new_val <= 0:
                    if new_val <= 0: # Make sure the length of the link is not negative
                        warnings.warn("Length of link must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                    elif "LINK_COM_POS_2" in new_vals.keys() and new_vals["LINK_COM_POS_2"] > new_val:
                        warnings.warn("Length of link must be greater than the position of its center of mass, link 2 length parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                    elif new_val < self.unwrapped.LINK_COM_POS_2:
                        warnings.warn("Length of link must be greater than the position of its center of mass, link 2 length parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue

                elif p == "LINK_MASS_1" and new_val <= 0:
                    warnings.warn("Mass of link 1 must be greater than zero, parameter not updated",ConstraintViolationWarning)
                    constraint_dict[p] = True

                elif p == "LINK_MASS_2" and new_val <= 0:
                    warnings.warn("Mass of link 2 must be greater than zero, parameter not updated",ConstraintViolationWarning)
                    constraint_dict[p] = True

                elif p == "LINK_COM_POS_1":
                    if new_val <= 0:
                        warnings.warn("Center of mass of link 1 must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                    elif "LINK_LENGTH_1" in new_vals.keys() and new_vals["LINK_LENGTH_1"] < new_val:
                        warnings.warn("Center of mass of link 1 must be less than the length of the link, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                    elif new_val > self.unwrapped.LINK_LENGTH_1:
                        warnings.warn("Center of mass of link 1 must be less than the length of the link, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                
                elif p == "LINK_COM_POS_2":
                    if new_val <= 0:
                        warnings.warn("Center of mass of link 2 must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                    elif "LINK_LENGTH_2" in new_vals.keys() and new_vals["LINK_LENGTH_2"] < new_val:
                        warnings.warn("Center of mass of link 2 must be less than the length of the link, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                    elif new_val > self.unwrapped.LINK_LENGTH_2:
                        warnings.warn("Center of mass of link 2 must be less than the length of the link, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                        continue
                                            
        elif self.unwrapped.__class__.__name__ == "MountainCarEnv": 
            for p,new_val in new_vals.items():
                constraint_dict[p] = False
                if p == "gravity":
                    if new_val <= 0:
                        warnings.warn("Gravity must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True

                if p == "power":
                    if new_val <= 0:
                        warnings.warn("Power must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True


        elif self.unwrapped.__class__.__name__ == "Continuous_MountainCarEnv":
            for p,new_val in new_vals.items():
                constraint_dict[p] = False
                if p == "power":
                    if new_val <= 0:
                        warnings.warn("Power must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True

        elif self.unwrapped.__class__.__name__ == "PendulumEnv": 
            for p,new_val in new_vals.items():
                constraint_dict[p] = False
                if p=="m":
                    if new_val <= 0:
                        warnings.warn("Mass of the pendulum must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                elif p=="l":
                    if new_val <= 0:
                        warnings.warn("Length of the pendulum must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True
                elif p=="g":
                    if new_val < 0:
                        warnings.warn("Gravity must be greater than or equal to zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True

                elif p=="dt":
                    if new_val <= 0:
                        warnings.warn("Time step must be greater than zero, parameter not updated",ConstraintViolationWarning)
                        constraint_dict[p] = True

        return constraint_dict

        
    def _dependency_resolver(self):
        """Check if the dependent parameters are updated accordingly.
        """
        if self.unwrapped.__class__.__name__ == "CartPoleEnv":
            if not (self.unwrapped.total_mass == self.unwrapped.masspole + self.unwrapped.masscart):
                setattr(self.unwrapped,"total_mass",self.unwrapped.masspole + self.unwrapped.masscart)
            if not (self.unwrapped.polemass_length == self.unwrapped.length * self.unwrapped.masspole):
                setattr(self.unwrapped,"polemass_length",self.unwrapped.length * self.unwrapped.masspole)

        elif self.unwrapped.__class__.__name__ == "AcrobotEnv": # no dependencies need to be updated 
            pass
        
        elif self.unwrapped.__class__.__name__ == "MountainCarEnv":
            pass
        
        elif self.unwrapped.__class__.__name__ == "PendulumEnv":
            pass

        elif self.unwrapped.__class__.__name__ == "Continuous_MountainCarEnv":
            pass
        



# nice 


if __name__ == "__main__":
    import ns_gym.base as base
    import ns_gym.update_functions as update_functions
    import ns_gym.schedulers as schedulers    
    import copy

    scheduler1 = schedulers.ContinuousScheduler()
    updateFn1 = update_functions.RandomWalk(scheduler=scheduler1)
    env = gym.make("CartPole-v1")
    params = {"force_mag":updateFn1}
    env = NSClassicControlWrapper(env,params)
    obs, info = env.reset()

    obs,reward, terminated, truncated, info = env.step(0)
    sim_env = copy.deepcopy(env)
    
    for _ in range(5):
        action = sim_env.action_space.sample()
        obs,reward, terminated, truncated, info = sim_env.step(action)
        obs,reward, terminated, truncated, info = env.step(action)
        print(f"sim_env {sim_env.unwrapped.force_mag}")
        print(f"env {env.unwrapped.force_mag}")
