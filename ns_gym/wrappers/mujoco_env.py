
import gymnasium as gym
import mujoco
import numpy as np
import warnings
from copy import deepcopy
from typing import Any
import ns_gym.base as base
import ns_gym.update_functions as update_functions
import ns_gym.schedulers as schedulers

class ConstraintViolationWarning(Warning):
    """Warning issued when a constraint in the application is violated."""
    pass

class MujocoWrapper(base.NSWrapper):

    def __init__(self, env: base.Env, tunable_params: dict, change_notification: bool = False, delta_change_notification: bool = False, in_sim_change: bool = False, **kwargs: Any):
        super().__init__(env=env,
                         tunable_params=tunable_params,
                         change_notification=change_notification,
                         delta_change_notification=delta_change_notification,
                         in_sim_change=in_sim_change,
                         **kwargs)
        self.t = 0
        self.delta_t = 1

        self.initial_params = {}
        self._param_map = self._create_param_map(tunable_params.keys())

        for key in tunable_params.keys():
            self.initial_params[key] = deepcopy(self._get_param_value(key))

    def _create_param_map(self, param_keys: list[str]) -> dict:
        """Creates a mapping from friendly parameter names to MuJoCo model attributes and indices."""
        param_map = {}
        model = self.unwrapped.model
        for key in param_keys:
            if hasattr(model.opt, key):
                param_map[key] = ('opt', key)
                continue

            parts = key.split('_')
            if len(parts) < 3:
                raise ValueError(f"Tunable parameter key '{key}' is not in the format 'type_name_param' (e.g., 'body_torso_mass').")
            
            obj_type, obj_name, param_name = parts[0], "_".join(parts[1:-1]), parts[-1]

            try:
                if obj_type == 'body':
                    idx = model.body(obj_name).id
                    attr_name = f"body_{param_name}"
                elif obj_type == 'geom':
                    idx = model.geom(obj_name).id
                    attr_name = f"geom_{param_name}"
                elif obj_type == 'dof':
                    idx = model.jnt(obj_name).dofadr[0]
                    attr_name = f"dof_{param_name}"
                else:
                    raise ValueError(f"Unsupported object type '{obj_type}'")

                assert hasattr(model, attr_name), f"Model does not have attribute {attr_name}"
                param_map[key] = (attr_name, idx)
                
            except Exception as e:
                raise ValueError(f"Could not parse key '{key}': {e}")

        return param_map

    def _get_param_value(self, key: str) -> Any:
        """Gets a parameter value from the MuJoCo model using the mapping."""
        location, identifier = self._param_map[key]
        if location == 'opt':
            return getattr(self.unwrapped.model.opt, identifier)
        else:
            return getattr(self.unwrapped.model, location)[identifier]

    def _set_param_value(self, key: str, value: Any):
        """Sets a parameter value in the MuJoCo model using the mapping."""
        location, identifier = self._param_map[key]
        if location == 'opt':
            np.copyto(getattr(self.unwrapped.model.opt, identifier), value)
        else:
            model_attr = getattr(self.unwrapped.model, location)
            model_attr[identifier] = value

    def _dependency_resolver(self):
        """Re-computes derived properties of the MuJoCo model after changes."""
        mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)

    def _constraint_checker(self, new_vals: dict) -> dict[str, bool]:
        """Checks if new parameter values violate physical constraints."""
        constraint_dict = {key: False for key in new_vals.keys()}
        for p, v in new_vals.items():
            if 'mass' in p and v <= 1e-6:
                warnings.warn(f"Mass for '{p}' must be positive, not updated.", ConstraintViolationWarning)
                constraint_dict[p] = True
            elif 'size' in p and np.any(np.array(v) <= 1e-6):
                 warnings.warn(f"Size for '{p}' must have positive elements, not updated.", ConstraintViolationWarning)
                 constraint_dict[p] = True
            elif 'damping' in p and v < 0:
                warnings.warn(f"Damping for '{p}' cannot be negative, not updated.", ConstraintViolationWarning)
                constraint_dict[p] = True
        return constraint_dict

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        """Applies physics changes and then steps the environment."""
        if self.is_sim_env and not self.in_sim_change:
            obs, reward, terminated, truncated, info = super().step(action, env_change=None, delta_change=None)
        else:
            env_change = {}
            delta_change = {}
            new_vals = {}
            
            for p, fn in self.tunable_params.items():
                cur_val = self._get_param_value(p)
                new_val, change_flag, delta = fn(cur_val, self.t)
                delta_change[p] = delta
                env_change[p] = change_flag
                new_vals[p] = new_val

            for k, v in self._constraint_checker(new_vals).items():
                if not v:  # If not violated, update the parameter
                    self._set_param_value(k, new_vals[k])
                else:
                    delta_change[k] = False
                    env_change[k] = False

            self._dependency_resolver()
            obs, reward, terminated, truncated, info = super().step(action, env_change=env_change, delta_change=delta_change)
        
        self.t += self.delta_t
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset environment and restore initial model parameters."""
        obs, info = super().reset(seed=seed, options=options)
        
        for k, v in self.initial_params.items():
            self._set_param_value(k, deepcopy(v))
        
        self._dependency_resolver()
        self.t = 0
        return obs, info


if __name__ == "__main__":
    env = gym.make("Ant-v5", render_mode="human", max_episode_steps=1000)

    # Define a real update function to make the Ant "floatier" over time
    scheduler = schedulers.ContinuousScheduler(start=100, end=150)
    # The step size will reduce gravity's pull each step
    updateFn = update_functions.StepWiseUpdate(scheduler, [np.array([0, 0, -9.8]), np.array([0, 0, -1000.0])])

    
    tunable_params = {"gravity": updateFn}
    ns_env = MujocoWrapper(env, tunable_params, change_notification=True)

    obs, info = ns_env.reset()
    print(f"Initial gravity: {ns_env._get_param_value('gravity')}")

    for i in range(100):
        action = ns_env.action_space.sample()
        obs, rew, done, truncated, info = ns_env.step(action)
        
        # Print the gravity every 20 steps to see it change
        if (i + 1) % 20 == 0:
            print(f"Gravity at step {i+1}: {np.round(ns_env._get_param_value('gravity'), 2)}")

        if done or truncated:
            obs, info = ns_env.reset()
            print("\n--- ENV RESET ---")
            print(f"Gravity after reset: {ns_env._get_param_value('gravity')}\n")

    ns_env.close()
