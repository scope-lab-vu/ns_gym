from typing import Any, Union
import gymnasium as gym
from copy import deepcopy
import warnings

import ns_gym.base as base


class ConstraintViolationWarning(Warning):
    """Warning issued when a constraint in the environment is violated."""

    pass


class NSClassicControlWrapper(base.NSWrapper):
    """A non-stationary wrapper for Gymnasium's Classic Control environments.

    Args:
        env (gym.Env): Base gym environment.
        tunable_params (dict[str,base.UpdateFn]): Dictionary of parameter names and their associated update functions.
        change_notification (bool, optional): Flag to indicate whether to notify the agent of changes in the environment. Defaults to False.
        delta_change_notification (bool, optional): Flag to indicate whether to notify the agent of changes in the transition function. Defaults to False.
        in_sim_change (bool, optional): Flag to allow environmental changes to occur in the 'planning' environment. Defaults to False.

    """

    def __init__(
        self,
        env,
        tunable_params,
        change_notification: bool = False,
        delta_change_notification: bool = False,
        in_sim_change: bool = False,
        **kwargs: Any,
    ):
        assert env.unwrapped.__class__.__name__ in base.TUNABLE_PARAMS.keys(), (
            f"{env.unwrapped.__class__.__name__} is not a supported environment"
        )
        super().__init__(
            env=env,
            tunable_params=tunable_params,
            change_notification=change_notification,
            delta_change_notification=delta_change_notification,
            in_sim_change=in_sim_change,
            **kwargs,
        )
        self.t = 0
        self.delta_t = 1

        self.initial_params = {}
        for key in tunable_params.keys():
            assert (
                key in base.TUNABLE_PARAMS[self.unwrapped.__class__.__name__].keys()
            ), (
                f"{key} is not a tunable parameter for {self.unwrapped.__class__.__name__}"
            )

            self.initial_params[key] = deepcopy(getattr(self.unwrapped, key))

    def step(self, action: Union[float, int]):
        """Step through environment and update environmental parameters

        Args:
            action (Union[float,int]): Action to take in environment

        Returns:
            tuple[dict[str, Any], base.Reward, bool, bool, dict[str, Any]]: NS-Gym Observation dictionary, reward, done flag, truncated flag, info dictionary
        """

        if self.is_sim_env and not self.in_sim_change:

            env_change = {p: 0 for p in self.tunable_params.keys()}
            delta_change = {p: 0.0 for p in self.tunable_params.keys()}
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change)
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

            self._dependency_resolver()
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )
            info["prob"] = 1.0

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset environment"""
        obs, info = super().reset(seed=seed, options=options)
        for k, v in self.initial_params.items():
            setattr(self.unwrapped, k, deepcopy(v))

        return obs, info

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
        assert self.has_reset, (
            "The environment must be reset before getting the planning environment."
        )
        if self.is_sim_env or self.change_notification:
            return deepcopy(self)
        elif not self.change_notification:
            planning_env = deepcopy(self)
            for k, v in self.initial_params.items():
                setattr(planning_env.unwrapped, k, deepcopy(v))
            return planning_env

    def __deepcopy__(self, memo):

        base_class_name = self.unwrapped.__class__.__name__
        
        # Standard Classic Control Mappings
        id_map = {
            "CartPoleEnv": "CartPole-v1",
            "MountainCarEnv": "MountainCar-v0",
            "Continuous_MountainCarEnv": "MountainCarContinuous-v0",
            "PendulumEnv": "Pendulum-v1",
            "AcrobotEnv": "Acrobot-v1",
        }

        # Fallback to spec.id only if we don't recognize the class
        base_id = id_map.get(base_class_name, self.unwrapped.spec.id)

        if self.unwrapped.__class__.__name__ in ["MountainCarEnv", "MountainCarContinuousEnv"]:
            warnings.warn(
                f"Deepcopy for {self.unwrapped.__class__.__name__} has a known "
                "issue with state divergence in the test suite. While parameter "
                "updates and notifications function correctly, be cautious if "
                "using `get_planning_env()` for simulation, as the test assertion "
                "`assert not np.array_equal(sim_obs['state'], obs['state'])` fails.",
                UserWarning
            )
        # Don't forward spec.kwargs — it contains NSWrapper kwargs
        # (change_notification, etc.) and gym.make() internal kwargs
        # (order_enforce, disable_env_checker) that aren't valid for the
        # base env constructor. Standard classic control envs don't need
        # any constructor kwargs.
        sim_env = gym.make(base_id)
        sim_env = NSClassicControlWrapper(
            sim_env,
            deepcopy(self.tunable_params),
            self.change_notification,
            self.delta_change_notification,
            self.in_sim_change,
            scalar_reward=self.scalar_reward,
        )
        sim_env.reset()
        sim_env.unwrapped.state = deepcopy(self.unwrapped.state)
        sim_env.t = deepcopy(self.t)
        for k, v in self.tunable_params.items():
            setattr(sim_env.unwrapped, k, deepcopy(getattr(self.unwrapped, k)))
        sim_env._dependency_resolver()
        sim_env.is_sim_env = True
        return sim_env
    

    def get_default_params(self):
        """Get dictionary of default parameters and their initial values"""
        return super().get_default_params()

    def _constraint_checker(self, new_vals) -> dict[str, bool]:
        """Check if the physical constraints of the environment are being violated, and all dependent parameters are updated accordingly.
        Checks evironment parameters after each update step. If a constraint is violated, the parameter does not update and a warning is issued.

        Args:
            new_vals (dict[str,float]): New value of the parameter.

        Returns:
            constraint_dict (dict[str,bool]): Dictionary of parameters and their constraint violation status. True is a constraint is violated, False otherwise.
        Note:
            Since each environement has different physical contraints, I can either create a new class for of each environment or just implement this method in the wrapper than check the base environment name.

        """
        constraint_dict: dict[str, bool] = {}

        if self.unwrapped.__class__.__name__ == "CartPoleEnv":
            for p, v in new_vals.items():
                constraint_dict[p] = False
                if p == "length" and v <= 0:
                    warnings.warn(
                        "Length of the pole cannot be negative, length not updated.",
                        ConstraintViolationWarning,
                    )
                    constraint_dict[p] = True
                elif p == "masscart" and v <= 0:
                    warnings.warn(
                        "Mass of the cart must be greater than zero, cart mass not updated",
                        ConstraintViolationWarning,
                    )
                    constraint_dict[p] = True
                elif p == "masspole" and v <= 0:
                    warnings.warn(
                        "Mass of the pole must be greater than zero, pole mass not updated",
                        ConstraintViolationWarning,
                    )
                    constraint_dict[p] = True
                elif p == "gravity" and v < 0:
                    warnings.warn(
                        "Gravity cannot be negative, gravity not updated",
                        ConstraintViolationWarning,
                    )
                    constraint_dict[p] = True
            return constraint_dict

        elif self.unwrapped.__class__.__name__ == "AcrobotEnv":
            for p, new_val in new_vals.items():
                constraint_dict[p] = False

                if p == "LINK_LENGTH_1":
                    if new_val <= 0:  # Make sure the length of the link is not negative
                        warnings.warn(
                            "Length of link must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue
                    elif (
                        "LINK_COM_POS_1" in new_vals.keys()
                        and new_vals["LINK_COM_POS_1"] > new_val
                    ):
                        warnings.warn(
                            "Length of link must be greater than the position of its center of mass, link 1 length parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue
                    elif new_val < self.unwrapped.LINK_COM_POS_1:
                        warnings.warn(
                            "Length of link must be greater than the position of its center of mass, link 1 length parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue

                elif p == "LINK_LENGTH_2" and new_val <= 0:
                    if new_val <= 0:  # Make sure the length of the link is not negative
                        warnings.warn(
                            "Length of link must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue
                    elif (
                        "LINK_COM_POS_2" in new_vals.keys()
                        and new_vals["LINK_COM_POS_2"] > new_val
                    ):
                        warnings.warn(
                            "Length of link must be greater than the position of its center of mass, link 2 length parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue
                    elif new_val < self.unwrapped.LINK_COM_POS_2:
                        warnings.warn(
                            "Length of link must be greater than the position of its center of mass, link 2 length parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue

                elif p == "LINK_MASS_1" and new_val <= 0:
                    warnings.warn(
                        "Mass of link 1 must be greater than zero, parameter not updated",
                        ConstraintViolationWarning,
                    )
                    constraint_dict[p] = True

                elif p == "LINK_MASS_2" and new_val <= 0:
                    warnings.warn(
                        "Mass of link 2 must be greater than zero, parameter not updated",
                        ConstraintViolationWarning,
                    )
                    constraint_dict[p] = True

                elif p == "LINK_COM_POS_1":
                    if new_val <= 0:
                        warnings.warn(
                            "Center of mass of link 1 must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue
                    elif (
                        "LINK_LENGTH_1" in new_vals.keys()
                        and new_vals["LINK_LENGTH_1"] < new_val
                    ):
                        warnings.warn(
                            "Center of mass of link 1 must be less than the length of the link, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue
                    elif new_val > self.unwrapped.LINK_LENGTH_1:
                        warnings.warn(
                            "Center of mass of link 1 must be less than the length of the link, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue

                elif p == "LINK_COM_POS_2":
                    if new_val <= 0:
                        warnings.warn(
                            "Center of mass of link 2 must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue
                    elif (
                        "LINK_LENGTH_2" in new_vals.keys()
                        and new_vals["LINK_LENGTH_2"] < new_val
                    ):
                        warnings.warn(
                            "Center of mass of link 2 must be less than the length of the link, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue
                    elif new_val > self.unwrapped.LINK_LENGTH_2:
                        warnings.warn(
                            "Center of mass of link 2 must be less than the length of the link, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                        continue

        elif self.unwrapped.__class__.__name__ == "MountainCarEnv":
            for p, new_val in new_vals.items():
                constraint_dict[p] = False
                if p == "gravity":
                    if new_val <= 0:
                        warnings.warn(
                            "Gravity must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True

                if p == "force":
                    if new_val <= 0:
                        warnings.warn(
                            "Force must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True

        elif self.unwrapped.__class__.__name__ == "Continuous_MountainCarEnv":
            for p, new_val in new_vals.items():
                constraint_dict[p] = False
                if p == "power":
                    if new_val <= 0:
                        warnings.warn(
                            "Power must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True

        elif self.unwrapped.__class__.__name__ == "PendulumEnv":
            for p, new_val in new_vals.items():
                constraint_dict[p] = False
                if p == "m":
                    if new_val <= 0:
                        warnings.warn(
                            "Mass of the pendulum must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                elif p == "l":
                    if new_val <= 0:
                        warnings.warn(
                            "Length of the pendulum must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True
                elif p == "g":
                    if new_val < 0:
                        warnings.warn(
                            "Gravity must be greater than or equal to zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True

                elif p == "dt":
                    if new_val <= 0:
                        warnings.warn(
                            "Time step must be greater than zero, parameter not updated",
                            ConstraintViolationWarning,
                        )
                        constraint_dict[p] = True

        return constraint_dict

    def _dependency_resolver(self):
        """Check if the dependent parameters are updated accordingly."""
        if self.unwrapped.__class__.__name__ == "CartPoleEnv":
            if not (
                self.unwrapped.total_mass
                == self.unwrapped.masspole + self.unwrapped.masscart
            ):
                setattr(
                    self.unwrapped,
                    "total_mass",
                    self.unwrapped.masspole + self.unwrapped.masscart,
                )
            if not (
                self.unwrapped.polemass_length
                == self.unwrapped.length * self.unwrapped.masspole
            ):
                setattr(
                    self.unwrapped,
                    "polemass_length",
                    self.unwrapped.length * self.unwrapped.masspole,
                )

        elif (
            self.unwrapped.__class__.__name__ == "AcrobotEnv"
        ):  # no dependencies need to be updated
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
    params = {"force_mag": updateFn1}
    env = NSClassicControlWrapper(env, params)
    obs, info = env.reset()

    obs, reward, terminated, truncated, info = env.step(0)
    sim_env = copy.deepcopy(env)

    for _ in range(5):
        action = sim_env.action_space.sample()
        obs, reward, terminated, truncated, info = sim_env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"sim_env {sim_env.unwrapped.force_mag}")
        print(f"env {env.unwrapped.force_mag}")
