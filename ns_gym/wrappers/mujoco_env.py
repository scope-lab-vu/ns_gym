import gymnasium as gym
import mujoco
import numpy as np
import warnings
from copy import deepcopy
from typing import Any
import ns_gym.base as base
import ns_gym.update_functions as update_functions
import ns_gym.schedulers as schedulers
from typing import Callable


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
        self.t = 0
        self.delta_t = 1

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
        """Re-computes derived properties of the MuJoCo model after changes."""
        mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)

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
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=None, delta_change=None
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

            self._dependency_resolver()
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )

        self.t += self.delta_t
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset environment and restore initial model parameters."""
        obs, info = super().reset(seed=seed, options=options)

        for k, v in self.initial_params.items():
            self._set_param_value(k, deepcopy(v))

        self._dependency_resolver()
        self.t = 0
        return obs, info


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
    mappings = {
        "AntEnv": {
            "gravity": (
                lambda env: env.model.opt.gravity,
                lambda env, val: np.copyto(env.model.opt.gravity, val),
            ),
            "torso_mass": (
                lambda env: env.model.body_mass[env.model.body("torso").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("torso").id, val
                ),
            ),
            "floor_friction": (
                lambda env: env.model.geom_friction[env.model.geom("floor").id, 0],
                # Corrected line below
                lambda env, val: env.model.geom_friction.__setitem__(
                    (env.model.geom("floor").id, 0), val
                ),
            ),
        },
        "HalfCheetahEnv": {
            "gravity": (
                lambda env: env.model.opt.gravity,
                lambda env, val: np.copyto(env.model.opt.gravity, val),
            ),
            "torso_mass": (
                lambda env: env.model.body_mass[env.model.body("torso").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("torso").id, val
                ),
            ),
            "bthigh_mass": (
                lambda env: env.model.body_mass[env.model.body("bthigh").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("bthigh").id, val
                ),
            ),
            "bshin_mass": (
                lambda env: env.model.body_mass[env.model.body("bshin").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("bshin").id, val
                ),
            ),
            "bfoot_mass": (
                lambda env: env.model.body_mass[env.model.body("bfoot").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("bfoot").id, val
                ),
            ),
            "fthigh_mass": (
                lambda env: env.model.body_mass[env.model.body("fthigh").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("fthigh").id, val
                ),
            ),
            "fshin_mass": (
                lambda env: env.model.body_mass[env.model.body("fshin").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("fshin").id, val
                ),
            ),
            "ffeet_mass": (
                lambda env: env.model.body_mass[env.model.body("ffoot").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("ffoot").id, val
                ),
            ),
            "floor_friction": (
                lambda env: env.model.geom_friction[env.model.geom("floor").id, 0],
                lambda env, val: env.model.geom_friction.__setitem__(
                    (env.model.geom("floor").id, 0), val
                ),
            ),
            "bthigh_damping": (
                lambda env: env.model.dof_damping[env.model.joint("bthigh").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("bthigh").id, val
                ),
            ),
            "bshin_damping": (
                lambda env: env.model.dof_damping[env.model.joint("bshin").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("bshin").id, val
                ),
            ),
            "bfoot_damping": (
                lambda env: env.model.dof_damping[env.model.joint("bfoot").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("bfoot").id, val
                ),
            ),
            "fthigh_damping": (
                lambda env: env.model.dof_damping[env.model.joint("fthigh").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("fthigh").id, val
                ),
            ),
            "fshin_damping": (
                lambda env: env.model.dof_damping[env.model.joint("fshin").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("fshin").id, val
                ),
            ),
            "ffeet_damping": (
                lambda env: env.model.dof_damping[env.model.joint("ffoot").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("ffoot").id, val
                ),
            ),
        },
        "HopperEnv": {
            "gravity": (
                lambda env: env.model.opt.gravity,
                lambda env, val: np.copyto(env.model.opt.gravity, val),
            ),
            "torso_mass": (
                lambda env: env.model.body_mass[env.model.body("torso").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("torso").id, val
                ),
            ),
            "thigh_mass": (
                lambda env: env.model.body_mass[env.model.body("thigh").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("thigh").id, val
                ),
            ),
            "leg_mass": (
                lambda env: env.model.body_mass[env.model.body("leg").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("leg").id, val
                ),
            ),
            "foot_mass": (
                lambda env: env.model.body_mass[env.model.body("foot").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("foot").id, val
                ),
            ),
            "floor_friction": (
                lambda env: env.model.geom_friction[env.model.geom("floor").id, 0],
                lambda env, val: env.model.geom_friction.__setitem__(
                    (env.model.geom("floor").id, 0), val
                ),
            ),
            "thigh_joint_damping": (
                lambda env: env.model.dof_damping[env.model.joint("thigh_joint").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("thigh_joint").id, val
                ),
            ),
            "leg_joint_damping": (
                lambda env: env.model.dof_damping[env.model.joint("leg_joint").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("leg_joint").id, val
                ),
            ),
            "foot_joint_damping": (
                lambda env: env.model.dof_damping[env.model.joint("foot_joint").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("foot_joint").id, val
                ),
            ),
        },
        "HumanoidEnv": {
            "gravity": (
                lambda env: env.model.opt.gravity,
                lambda env, val: np.copyto(env.model.opt.gravity, val),
            ),
            "torso_mass": (
                lambda env: env.model.body_mass[env.model.body("torso").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("torso").id, val
                ),
            ),
            "lwaist_mass": (
                lambda env: env.model.body_mass[env.model.body("lwaist").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("lwaist").id, val
                ),
            ),
            "pelvis_mass": (
                lambda env: env.model.body_mass[env.model.body("pelvis").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("pelvis").id, val
                ),
            ),
            "right_thigh_mass": (
                lambda env: env.model.body_mass[env.model.body("right_thigh").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("right_thigh").id, val
                ),
            ),
            "left_thigh_mass": (
                lambda env: env.model.body_mass[env.model.body("left_thigh").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("left_thigh").id, val
                ),
            ),
            "right_shin_mass": (
                lambda env: env.model.body_mass[env.model.body("right_shin").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("right_shin").id, val
                ),
            ),
            "left_shin_mass": (
                lambda env: env.model.body_mass[env.model.body("left_shin").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("left_shin").id, val
                ),
            ),
            "right_foot_mass": (
                lambda env: env.model.body_mass[env.model.body("right_foot").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("right_foot").id, val
                ),
            ),
            "left_foot_mass": (
                lambda env: env.model.body_mass[env.model.body("left_foot").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("left_foot").id, val
                ),
            ),
            "right_upper_arm_mass": (
                lambda env: env.model.body_mass[env.model.body("right_upper_arm").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("right_upper_arm").id, val
                ),
            ),
            "left_upper_arm_mass": (
                lambda env: env.model.body_mass[env.model.body("left_upper_arm").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("left_upper_arm").id, val
                ),
            ),
            "right_lower_arm_mass": (
                lambda env: env.model.body_mass[env.model.body("right_lower_arm").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("right_lower_arm").id, val
                ),
            ),
            "left_lower_arm_mass": (
                lambda env: env.model.body_mass[env.model.body("left_lower_arm").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("left_lower_arm").id, val
                ),
            ),
            "floor_friction": (
                lambda env: env.model.geom_friction[env.model.geom("floor").id, 0],
                lambda env, val: env.model.geom_friction.__setitem__(
                    (env.model.geom("floor").id, 0), val
                ),
            ),
            "right_knee_damping": (
                lambda env: env.model.dof_damping[env.model.joint("right_knee").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("right_knee").id, val
                ),
            ),
            "left_knee_damping": (
                lambda env: env.model.dof_damping[env.model.joint("left_knee").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("left_knee").id, val
                ),
            ),
            "right_elbow_damping": (
                lambda env: env.model.dof_damping[env.model.joint("right_elbow").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("right_elbow").id, val
                ),
            ),
            "left_elbow_damping": (
                lambda env: env.model.dof_damping[env.model.joint("left_elbow").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("left_elbow").id, val
                ),
            ),
        },
        "InvertedPendulumEnv": {
            "gravity": (
                lambda env: env.model.opt.gravity,
                lambda env, val: np.copyto(env.model.opt.gravity, val),
            ),
            "pole_mass": (
                lambda env: env.model.body_mass[env.model.body("pole").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("pole").id, val
                ),
            ),
            "cart_mass": (
                lambda env: env.model.body_mass[env.model.body("cart").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("cart").id, val
                ),
            ),
            "rail_friction": (
                lambda env: env.model.geom_friction[env.model.geom("rail").id, 0],
                lambda env, val: env.model.geom_friction.__setitem__(
                    (env.model.geom("rail").id, 0), val
                ),
            ),
        },
        "InvertedDoublePendulumEnv": {
            "gravity": (
                lambda env: env.model.opt.gravity,
                lambda env, val: np.copyto(env.model.opt.gravity, val),
            ),
            "cart_mass": (
                lambda env: env.model.body_mass[env.model.body("cart").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("cart").id, val
                ),
            ),
            "pole1_mass": (
                lambda env: env.model.body_mass[env.model.body("pole").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("pole").id, val
                ),
            ),
            "pole2_mass": (
                lambda env: env.model.body_mass[env.model.body("pole2").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("pole2").id, val
                ),
            ),
            "floor_friction": (
                lambda env: env.model.geom_friction[env.model.geom("rail").id, 0],
                lambda env, val: env.model.geom_friction.__setitem__(
                    (env.model.geom("rail").id, 0), val
                ),
            ),
            "slider_damping": (
                lambda env: env.model.dof_damping[env.model.joint("slider").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("slider").id, val
                ),
            ),
            "hinge1_damping": (
                lambda env: env.model.dof_damping[env.model.joint("hinge").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("hinge").id, val
                ),
            ),
            "hinge2_damping": (
                lambda env: env.model.dof_damping[env.model.joint("hinge2").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("hinge2").id, val
                ),
            ),
        },
        "ReacherEnv": {
            # "gravity": (
            #     lambda env: env.model.opt.gravity,
            #     lambda env, val: np.copyto(env.model.opt.gravity, val),
            # ),
            "body0_mass": (
                lambda env: env.model.body_mass[env.model.body("body0").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("body0").id, val
                ),
            ),
            "body1_mass": (
                lambda env: env.model.body_mass[env.model.body("body1").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("body1").id, val
                ),
            ),
            # "ground_friction": (
            #     lambda env: env.model.geom_friction[env.model.geom("ground").id, 0],
            #     lambda env, val: env.model.geom_friction.__setitem__(
            #         (env.model.geom("ground").id, 0), val
            #     ),
            # ),
            "joint0_damping": (
                lambda env: env.model.dof_damping[env.model.joint("joint0").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("joint0").id, val
                ),
            ),
            "joint1_damping": (
                lambda env: env.model.dof_damping[env.model.joint("joint1").id],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("joint1").id, val
                ),
            ),
        },
        "SwimmerEnv": {
            # "gravity": (
            #     lambda env: env.model.opt.gravity,
            #     lambda env, val: np.copyto(env.model.opt.gravity, val),
            # ),
            "body_mid_mass": (
                lambda env: env.model.body_mass[env.model.body("mid").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("mid").id, val
                ),
            ),
            # "geom_floor_friction": (
            #     lambda env: env.model.geom_friction[env.model.geom("floor").id, 0],
            #     lambda env, val: env.model.geom_friction.__setitem__(
            #         (env.model.geom("floor").id, 0), val
            #     ),
            # ),
        },
        "PusherEnv": {
            "gravity": (
                lambda env: env.model.opt.gravity,
                lambda env, val: np.copyto(env.model.opt.gravity, val),
            ),
            "r_shoulder_pan_link_mass": (
                lambda env: env.model.body_mass[
                    env.model.body("r_shoulder_pan_link").id
                ],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("r_shoulder_pan_link").id, val
                ),
            ),
            "r_shoulder_lift_link_mass": (
                lambda env: env.model.body_mass[
                    env.model.body("r_shoulder_lift_link").id
                ],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("r_shoulder_lift_link").id, val
                ),
            ),
            "r_upper_arm_link_mass": (
                lambda env: env.model.body_mass[env.model.body("r_upper_arm_link").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("r_upper_arm_link").id, val
                ),
            ),
            "r_forearm_link_mass": (
                lambda env: env.model.body_mass[env.model.body("r_forearm_link").id],
                lambda env, val: env.model.body_mass.__setitem__(
                    env.model.body("r_forearm_link").id, val
                ),
            ),
            "r_shoulder_pan_joint_damping": (
                lambda env: env.model.dof_damping[
                    env.model.joint("r_shoulder_pan_joint").id
                ],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("r_shoulder_pan_joint").id, val
                ),
            ),
            "r_shoulder_lift_joint_damping": (
                lambda env: env.model.dof_damping[
                    env.model.joint("r_shoulder_lift_joint").id
                ],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("r_shoulder_lift_joint").id, val
                ),
            ),
            "r_elbow_flex_joint_damping": (
                lambda env: env.model.dof_damping[
                    env.model.joint("r_elbow_flex_joint").id
                ],
                lambda env, val: env.model.dof_damping.__setitem__(
                    env.model.joint("r_elbow_flex_joint").id, val
                ),
            ),
        },
    }

    # HumanoidStandupEnv uses the same XML and attributes as HumanoidEnv
    mappings["HumanoidStandupEnv"] = mappings["HumanoidEnv"]

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
