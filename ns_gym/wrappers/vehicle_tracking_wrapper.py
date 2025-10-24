import ns_gym
import ns_gym.base as base
from copy import deepcopy
import numpy as np
from collections import defaultdict
from gymnasium import gym   

class VehicleTrackingWrapper(base.NSWrapper):
    """A wrapper for the Vehicle Tracking environment that allows for dynamic tuning of environment parameters during simulation.

    Args:
        env: The Vehicle Tracking environment to be wrapped.
        tunable_params: A dictionary mapping parameter names to functions that determine how to update them.
        change_notification (bool): If True, the environment will notify when a parameter has changed.
        delta_change_notification (bool): If True, the environment will notify the magnitude of the change in parameters.
        in_sim_change (bool): If True, allows parameter changes during simulation.
    """
    def __init__(self,
                 env,
                 tunable_params,
                 change_notification: bool = False,
                 delta_change_notification: bool = False,
                 in_sim_change: bool = False,):
        super().__init__(env, tunable_params, change_notification, delta_change_notification, in_sim_change)


        initial_params = {}

        for param in tunable_params:
            initial_params[param] = deepcopy(self._get_param_value(param))

        self.initial_params = initial_params

    def step(self, action):
        """Take a step in the environment, updating tunable parameters as necessary.
        """

        if self.is_sim_env and not self.in_sim_change:
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=None, delta_change=None
            )
        else:
            env_change = {}
            delta_change = {}
            new_vals = {}
            for p, fn in self.tunable_params.items():
                cur_val = getattr(self.unwrapped, p)
                new_val, change_flag, delta = fn(cur_val, self.t)
                delta_change[p] = delta
                env_change[p] = change_flag
                new_vals[p] = new_val

            for k, v in self._constraint_checker(new_vals).items():
                if not v:  # If the constraint is not violated, update the parameter
                    setattr(self.unwrapped, k, new_vals[k])
                else:
                    delta_change[k] = 0.0
                    env_change[k] = 0

            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )

        return obs, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        """Reset the environment, restoring initial parameter values.
        """
        obs, info = super().reset(seed=seed, options=options)

        for k, v in self.initial_params.items():
            setattr(self.unwrapped, k, deepcopy(v))

        return obs, info

    
    def _get_param_value(self, param_name):
        return getattr(self.env.unwrapped, param_name)
    
    def _constraint_checker(self, new_vals):

        constraint_violation_lookup_table = {
           "num_pursuers": lambda x: x >= 1,
           "fov_distance": lambda x: x >= 0.0,
           "fov_angle": lambda x: 0.0 <= x <= 2 * np.pi,
           "game_mode": lambda x: x in ["normal", "hard"],
           "is_evader_always_observable": lambda x: isinstance(x, bool),
           "allow_diagonal_movement": lambda x: isinstance(x, bool),
           "goal_locations": lambda x: (0 <= x[0] <= self.unwrapped.map_size[0]) and (0 <= x[1] <= self.unwrapped.map_size[1])
        }
        constraint_dict: dict[str, bool] = defaultdict(False)

        for p, val in new_vals.items():
            constraint_dict[p] = constraint_violation_lookup_table[p](val)

        return constraint_dict


    def get_planning_env(self):
        if self.is_sim_env or self.change_notification:
            return deepcopy(self)
        elif not self.change_notification:
            planning_env = deepcopy(self)
            for k, v in self.initial_params.items():
                setattr(planning_env.unwrapped, k, deepcopy(v))
            return planning_env


    def __deepcopy__(self, memo):
        env_kwargs = self.unwrapped.spec.kwargs
        sim_env = gym.make(self.unwrapped.spec.id,**env_kwargs)
        sim_env = VehicleTrackingWrapper(
            sim_env,
            deepcopy(self.tunable_params),
            self.change_notification,
            self.delta_change_notification,
            self.in_sim_change,
        )
        sim_env.reset()
        sim_env.unwrapped.state = deepcopy(self.unwrapped.state)
        sim_env.t = deepcopy(self.t)
        for k, v in self.tunable_params.items():
            setattr(sim_env.unwrapped, k, deepcopy(getattr(self.unwrapped, k)))
        sim_env._dependency_resolver()
        sim_env.is_sim_env = True
        return sim_env
    




    

    


    


        

