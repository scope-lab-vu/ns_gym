import gymnasium as gym
import mujoco
import numpy as np
import warnings
from copy import deepcopy
from typing import Any, Callable
import ns_gym.base as base
import ns_gym.update_functions as update_functions
import ns_gym.schedulers as schedulers


class ConstraintViolationWarning(Warning):
    """Warning issued when a constraint in the application is violated."""

    pass


class MujocoWrapper(base.NSWrapper):
    def __init__(
        self,
        env: base.Env,
        tunable_params: dict,
        change_notification: bool = False,
        delta_change_notification: bool = False,
        in_sim_change: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            env=env,
            tunable_params=tunable_params,
            change_notification=change_notification,
            delta_change_notification=delta_change_notification,
            in_sim_change=in_sim_change,
            **kwargs,
        )
        self._scratch_data = None
        self._accessors = {}
        for key in tunable_params.keys():
            self._accessors[key] = param_look_up(
                self.unwrapped.__class__.__name__, key
            )[0]

        self.initial_params = {}
        for key in tunable_params.keys():
            self.initial_params[key] = deepcopy(self._get_param_value(key))

    def _get_param_value(self, key: str) -> Any:
        """Gets a parameter value by calling its specific getter function."""
        getter, _ = self._accessors[key]
        return getter(self.unwrapped)

    def _set_param_value(self, key: str, value: Any):
        """Sets a parameter value by calling its specific setter function."""
        _, setter = self._accessors[key]
        setter(self.unwrapped, value)

    def _dependency_resolver(self):
        """Recompute the model constants that depend on tunable parameters.

        ``mj_setConst`` refreshes derived model constants such as
        ``body_subtreemass`` after a mass change. It uses an ``MjData`` as its
        workspace and leaves that data in the reference configuration, so it is
        run on a scratch ``MjData`` rather than on the live one.

        The live simulation data is deliberately left untouched. The base
        Gymnasium envs read body positions from it to compute rewards (Ant and
        Humanoid measure velocity from the position before and after the step),
        so refreshing it here would make a wrapped env pay different rewards
        than the unwrapped one even when no parameter changes.
        """
        model = self.unwrapped.model
        if self._scratch_data is None:
            self._scratch_data = mujoco.MjData(model)
        mujoco.mj_setConst(model, self._scratch_data)

    def _constraint_checker(self, new_vals: dict) -> dict[str, bool]:
        """Checks if new parameter values violate physical constraints."""
        constraint_dict = {key: False for key in new_vals.keys()}
        for p, v in new_vals.items():
            if "mass" in p and v <= 1e-6:
                warnings.warn(
                    f"Mass for '{p}' must be positive, not updated.",
                    ConstraintViolationWarning,
                )
                constraint_dict[p] = True
            elif "size" in p and np.any(np.array(v) <= 1e-6):
                warnings.warn(
                    f"Size for '{p}' must have positive elements, not updated.",
                    ConstraintViolationWarning,
                )
                constraint_dict[p] = True
            elif "damping" in p and v < 0:
                warnings.warn(
                    f"Damping for '{p}' cannot be negative, not updated.",
                    ConstraintViolationWarning,
                )
                constraint_dict[p] = True

        return constraint_dict

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        """Applies physics changes and then steps the environment."""
        if self.is_sim_env and not self.in_sim_change:

            env_change = {p: 0 for p in self.tunable_params.keys()}
            delta_change = {p: 0.0 for p in self.tunable_params.keys()}
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )
        else:
            env_change = {}
            delta_change = {}
            new_vals = {}

            for p, fn in self.tunable_params.items():
                if p == "gravity":
                    # special handeling for gravity
                    cur_val = self._get_param_value(p)
                    val_to_update = cur_val[-1]  # Get the z-component of gravity
                    new_val, change_flag, delta = fn(val_to_update, self.t)
                    delta_change[p] = delta
                    env_change[p] = change_flag
                    new_vals[p] = np.array([cur_val[0], cur_val[1], new_val])
                else:
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

            if any(env_change.values()):
                self._dependency_resolver()
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Restore initial model parameters, then reset the environment.

        The parameters are restored first so the base env computes its reset
        observation with them. Some observations depend on the parameters
        directly (Humanoid's includes body masses).
        """
        if not self.persistent_params:
            for k, v in self.initial_params.items():
                self._set_param_value(k, deepcopy(v))
            self._dependency_resolver()

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def get_planning_env(self):
        """Return a copy of the environment for planning.

        NOTE:
            - If the environment is a simulation environment, the function returns a deepcopy of the simulation environment.
            - If change notification is enabled, the function returns a deepcopy of the current environment because the decision making agent needs to be aware of the changes in the environment.
            - If change notification is disabled, the function returns a deepcopy of the environment with the initial parameters.
        """
        assert self.has_reset, (
            "The environment must be reset before getting the planning environment."
        )
        if self.is_sim_env or self.delta_change_notification:
            return deepcopy(self)
        elif not self.delta_change_notification:
            planning_env = deepcopy(self)
            for k, v in self.initial_params.items():
                planning_env._set_param_value(k, deepcopy(v))
            planning_env._dependency_resolver()
            return planning_env

    def __deepcopy__(self, memo):
        base_class_name = self.unwrapped.__class__.__name__

        id_map = {
            "AntEnv": "Ant-v5",
            "HalfCheetahEnv": "HalfCheetah-v5",
            "HopperEnv": "Hopper-v5",
            "HumanoidEnv": "Humanoid-v5",
            "HumanoidStandupEnv": "HumanoidStandup-v5",
            "InvertedPendulumEnv": "InvertedPendulum-v5",
            "InvertedDoublePendulumEnv": "InvertedDoublePendulum-v5",
            "ReacherEnv": "Reacher-v5",
            "SwimmerEnv": "Swimmer-v5",
            "PusherEnv": "Pusher-v5",
        }

        # Rebuilt with the base env's own id and constructor arguments
        # (``terminate_when_unhealthy``, ``xml_file``, ...); the id map is the
        # fallback for an env that was not created through gym.make.
        sim_env = self._make_base_env(id_map.get(base_class_name))
        sim_env = MujocoWrapper(
            sim_env,
            deepcopy(self.tunable_params),
            self.change_notification,
            self.delta_change_notification,
            self.in_sim_change,
            scalar_reward=self.scalar_reward,
            persistent_params=self.persistent_params,
        )
        sim_env.reset()

        # Copy current tunable parameter values into the new model
        for k in self.tunable_params:
            sim_env._set_param_value(k, deepcopy(self._get_param_value(k)))
        sim_env._dependency_resolver()

        # Clone the whole simulation, not only the integration state: the base
        # env computes rewards from derived quantities (body positions), so the
        # copy must hold the same ones to continue exactly like the original.
        _copy_simulation(
            self.unwrapped.model,
            self.unwrapped.data,
            sim_env.unwrapped.model,
            sim_env.unwrapped.data,
        )

        # Copy wrapper state
        sim_env.t = deepcopy(self.t)
        sim_env.is_sim_env = True
        sim_env._reseed_planning_env_rngs()
        return sim_env


def _copy_simulation(src_model, src_data, dst_model, dst_data) -> None:
    """Copy a running MuJoCo simulation into another ``MjData`` of the same model.

    The Python bindings have no ``mj_copyData`` for two separately built
    models, so this is done in two parts:

    1. The integration state (time, qpos, qvel, act, warm start) through
       MuJoCo's state API.
    2. Every fixed-size array of ``MjData``. This carries the inputs that
       persist between steps (``ctrl``, ``xfrc_applied``, mocap poses) and the
       derived quantities as the last ``mj_step`` left them (``xpos``,
       ``cfrc_ext``...), which the Gymnasium envs read for rewards and
       observations.

    Constraint and island arrays are skipped: their size depends on the
    current contacts, and the next ``mj_step`` rebuilds them from the state.
    """
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(src_model, spec))
    mujoco.mj_getState(src_model, src_data, state, spec)
    mujoco.mj_setState(dst_model, dst_data, state, spec)

    for name in dir(src_data):
        if name.startswith(("_", "efc_")) or "island" in name:
            continue
        src_field = getattr(src_data, name, None)
        if not isinstance(src_field, np.ndarray):
            continue
        dst_field = getattr(dst_data, name, None)
        if isinstance(dst_field, np.ndarray) and dst_field.shape == src_field.shape:
            np.copyto(dst_field, src_field)


def param_look_up(env_name: str, tunable_param: str) -> tuple[Callable, Callable]:
    """Helper function to grab setter and getter functions for various MuJoCo environments.
    Maps friendly parameter names to MuJoCo model attributes and indices.

    Args:
        env_name (str): Name of the MuJoCo environment class (e.g., "AntEnv").
        tunable_param (str): Friendly name of the tunable parameter (e.g., "torso_mass").
    Returns:
        tuple[Callable, Callable]: A tuple containing two callables:
            - A getter function that takes an env instance and returns the parameter value.
            - A setter function that takes an env instance and a value, and sets the parameter.

    """
    mappings = base.MUJOCO_GETTERS

    if env_name in mappings:
        env_mapping = mappings[env_name]
        if tunable_param in env_mapping:
            return [env_mapping[tunable_param]]
        else:
            raise ValueError(
                f"Parameter '{tunable_param}' not recognized for environment '{env_name}'."
            )
    else:
        raise ValueError(f"Environment '{env_name}' not recognized or supported.")


if __name__ == "__main__":
    env = gym.make("Ant-v5", render_mode="human", max_episode_steps=1000)

    # Define a real update function to make the Ant "floatier" over time
    scheduler = schedulers.ContinuousScheduler(start=10, end=1000)
    # The step size will reduce gravity's pull each step
    updateFn = update_functions.StepWiseUpdate(
        scheduler, [np.array([0, 0, -9.8]), np.array([0, 0, -1000.0])]
    )

    tunable_params = {"gravity": updateFn}
    ns_env = MujocoWrapper(env, tunable_params, change_notification=True)

    obs, info = ns_env.reset()
    print(f"Initial gravity: {ns_env._get_param_value('gravity')}")

    for i in range(100):
        action = ns_env.action_space.sample()
        obs, rew, done, truncated, info = ns_env.step(action)

        # Print the gravity every 2 steps to see it change
        if (i + 1) % 2 == 0:
            print(
                f"Gravity at step {i + 1}: {np.round(ns_env._get_param_value('gravity'), 2)}"
            )

        if done or truncated:
            obs, info = ns_env.reset()
            print("\n--- ENV RESET ---")
            print(f"Gravity after reset: {ns_env._get_param_value('gravity')}\n")

    ns_env.close()
