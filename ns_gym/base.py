import numpy as np
from abc import ABC
from typing import Union, Any, Type
from dataclasses import dataclass
from gymnasium import Env, Wrapper
import gymnasium as gym
import ns_gym.utils as utils
import copy


"""
Some core types 
"""


# @dataclass(frozen=True)
# class Observation:
#     """Observation dataclass type. This is the output of the step function in the environment."""

#     #: The state of the environment
#     state: Union[np.ndarray, int]

#     #: A dictionary of boolean flags indicating what param of the environment has changed.
#     env_change: Union[dict[str, bool], None]

#     #: The amount of change in the transition function of the environment.
#     delta_change: Union[dict[float, float], float, None]

#     #: The relative time of the observation since the start of the environment episode.
#     relative_time: Union[int, float]


@dataclass(frozen=True)
class Reward:
    """Reward dataclass type. This is the output of the step function in the environment."""

    #: The reward received from the environment
    reward: Union[int, float]

    #: A dictionary of boolean flags indicating what param of the environment has changed.
    env_change: dict[str, bool]

    #: The change in the reward function of the environment.
    delta_change: Union[float, None]

    #: The relative time of the observation since the start of the environment episode.
    relative_time: Union[int, float]


class Scheduler(ABC):
    """Base class for scheduler functions. This class is used to determine when to update a parameter in the environment.

    Start and end times inclusive.
    """

    def __init__(self, start=0, end=np.inf) -> None:
        """Initialize the scheduler.

        Args:
            start (int): The start time of the scheduler. Defaults to 0.
            end (int): The end time of the scheduler. Defaults to np.inf.
        """
        super().__init__()
        self.start = start
        self.end = end

    def __call__(self, t: int) -> bool:
        """Call method to determine whether to update the parameter or not. Subclasses must implement this method.
        Args:
            t (int): MDP timestep

        Returns:
            bool: Boolean flag indicating whether to update the parameter or not.
        """
        return NotImplementedError("Subclasses must implement this method")


class UpdateFn(ABC):
    """Base class for update functions that update a single parameter. Updates a scalar parameter

    Overview:
        Instances of this class (and all subclasses) are **callable** and should be used to apply an update to a parameter. When an instance is called it executes the update logic defined in the subclass's `_update` method. The `__call__` method checks with the provided `Scheduler` to determine if an update should occur at the current time step. If an update is warranted, it invokes the `_update` method to modify the parameter and calculates the change in value.

    Args:
        scheduler (Scheduler): scheduler object that determines when to update the parameter
        scheduler (Scheduler): scheduler object that determines when to update the parameter

    Attributes:
        prev_param: The previous parameter value
        prev_time: The previous time the parameter was updated

    """

    def __init__(self, scheduler: Scheduler) -> None:
        """ """

        assert isinstance(scheduler, Scheduler), (
            f"Expected scheduler to be a subclass of Scheduler, got {type(scheduler)}"
        )
        self.scheduler = scheduler
        self.prev_param = None
        self.prev_time = -1

    def __call__(self, param: Any, t: Union[int, float]) -> tuple[Any, int, float]:
        """Update the parameter if the scheduler returns True

        Args:
            param (Any): The parameter to be updated
            t (Union[int,float]): The current time step

        Returns:
            Union[int, float]: The updated parameter
            int: Binary flag indicating whether the parameter was updated or not, 1 means updated, 0 means not updated
            float: The amount of change in the parameter
        """
        assert isinstance(t, (int, float)), (
            f"Expected t to be an int or float, got {type(t)}, Arrays operations need to inherit from UpdateDistributionFn"
        )
        if self.scheduler(t):
            updated_param = self._update(copy.copy(param), t)

            delta_change = self._get_delta_change(param, updated_param, t)
            self.prev_param = param
            self.prev_time = t
            return (updated_param, 1, delta_change)
        else:
            self.prev_param = param
            self.prev_time = t
            return (param, 0, 0.0)

    def _update(self, param: Any, t: int) -> Any:
        """Update the parameter. Subclasses must implement this method. Called by the __call__ method if the scheduler returns True.

        Args:
            param (Any): The parameter to be updated
            t (int): The current time step

        Returns:
            Any: The updated parameter
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_delta_change(self, param: Any, updated_param: Any, t: int) -> float:
        """Get the amount of change in the parameter. Default implementation is to return the absolute difference between the updated and previous parameter.

        Args:
            param (Any): The parameter to be updated
            updated_param (Any): The updated parameter
            t (int): The current time step
        Returns:
            float: The amount of change in the parameter
        """
        return updated_param - param


class UpdateDistributionFn(UpdateFn):
    """Base class for all update functions that update a distribution represented as a list"""

    def __call__(self, param: Any, t: Union[int, float]) -> Any:
        assert isinstance(param, list), f"param must be a list, got {type(param)}"
        return super().__call__(param, t)

    def _get_delta_change(self, param: Any, updated_param: Any, t: int) -> float:
        """We will use the Wasserstein distance to measure the amount of change in the distribution.

        Args:
            param (Any): The parameter to be updated
            t (int): The current time step

        Returns:
            float: Amount of change in the distribution

        """
        return utils.wasserstein_distance(param, updated_param)


class NSWrapper(Wrapper):
    """Base class for non-stationary wrappers

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

    def __init__(
        self,
        env: Type[Env],
        tunable_params: dict[str, Union[Type[UpdateFn], Type[UpdateDistributionFn]]],
        change_notification: bool = False,
        delta_change_notification: bool = False,
        in_sim_change: bool = False,
        scalar_reward: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            env (Env): Gym environment
            tunable_params (dict[str,Union[Type[UpdateFn],Type[UpdateDistributionFn]]): Dictionary of parameter names and their associated update functions.
            change_notification (bool): Sets a basic notification level. Returns a boolean flag to indicate whether to notify the agent of changes in the environment. Defaults to False.
            delta_change_notification (bool): Sets detailed notification levle. Returns Flag to indicate whether to notify the agent of changes in the transition function. Defaults to False.
            in_sim_change (bool): Flag to indicate whether to allow changes in the environment during simulation (e.g MCTS rollouts). Defaults to False.
            scalar_reward (bool): If True, step() returns a scalar reward. If False, step() returns a Reward dataclass containing the reward, env_change, delta_change, and relative_time. Defaults to True.

        Attributes:
            frozen (bool): Flag to indicate whether the environment is frozen or not.
            is_sim_env (bool): Flag to indicate whether the environment is a simulation environment or not.

        """
        Wrapper.__init__(self, env)

        self.scalar_reward = scalar_reward

        if delta_change_notification:
            assert change_notification, (
                "If change_notification is True, delta_change_notification must be True"
            )

        assert set(tunable_params.keys()) <= set(
            TUNABLE_PARAMS.get(self.unwrapped.__class__.__name__, {}).keys()
        ), (
            f"Tunable parameters {list(tunable_params.keys())} not all in default tunable parameters {list(TUNABLE_PARAMS.get(self.unwrapped.__class__.__name__, {}).keys())} for environment {self.unwrapped.__class__.__name__}"
        )

        self.tunable_params = tunable_params

        self.init_initial_params = copy.deepcopy(self.tunable_params)
        self.change_notification = change_notification
        self.delta_change_notification = delta_change_notification
        self.in_sim_change = in_sim_change
        self.frozen = False
        self.is_sim_env = False
        self.t = 0
        self.has_reset = False

        env_change_space = gym.spaces.Dict(
            {param_name: gym.spaces.Discrete(2) for param_name in tunable_params.keys()}
        )
        delta_change_space = gym.spaces.Dict(
            {
                param_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=())
                for param_name in tunable_params.keys()
            }
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": self.unwrapped.observation_space,
                "env_change": env_change_space,
                "delta_change": delta_change_space,
                "relative_time": gym.spaces.Box(low=0, high=np.inf, shape=()),
            }
        )

        setattr(self.unwrapped, "get_planning_env", self.get_planning_env)

    def step(
        self, action: Any, env_change: dict[str, Union[bool, int]], delta_change: dict[str, Union[bool, int, float]]
    ):
        """Step function for the environment. Augments observations and rewards with additional information about changes in the environment and transition function.

        Subclasses of this class will handle the actual environment dynamics and updating of parameters. This base class handles the notification mechanism that emulates the run-time monitor and model updater components of the decision-making infrastructure. The subclass must call this function via super().step(action, env_change, delta_change).

        Args:
            action (int): Action taken by the agent
            env_change (dict[str,bool]): Environment change flags. Keys are parameter names and values are boolean flags indicating whether the parameter has changed.
            delta_change (dict[str,bool]): The amount of change a parameter has undergone. Keys are parameter names and values are the amount of change.

        Returns:
            tuple[observation, Type[Reward], bool, bool, dict[str, Any]]: observation, reward, termination flag, truncation flag, and additional information.

        """

        state, reward, terminated, truncated, info = super().step(action)
        self.t += 1

        default_env_change = {p: 0 for p in self.tunable_params.keys()}
        default_delta_change = {p: 0.0 for p in self.tunable_params.keys()}

        calculated_env_change = {k: int(v) for k, v in env_change.items()}

        calculated_delta_change = {k: float(v) for k, v in delta_change.items()}

        if (
            not self.change_notification
            or self.frozen
            or (self.is_sim_env and not self.in_sim_change)
        ):
            final_env_change = default_env_change
        else:
            
            final_env_change = {**default_env_change, **calculated_env_change}

        if (
            not self.delta_change_notification
            or self.frozen
            or (self.is_sim_env and not self.in_sim_change)
        ):
            final_delta_change = default_delta_change
        else:
            
            final_delta_change = {**default_delta_change, **calculated_delta_change}

        obs = {
            "state": state,
            "env_change": final_env_change,
            "delta_change": final_delta_change,
            "relative_time": self.t,
        }

        if self.scalar_reward:
            rew = reward
        else:
            rew = Reward(
                reward=reward,
                env_change=final_env_change,
                delta_change=final_delta_change,
                relative_time=self.t,
            )

        info["Ground Truth Env Change"] = calculated_env_change
        info["Ground Truth Delta Change"] = calculated_delta_change

        return obs, rew, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset function for the environment. Resets the environment to its initial state and resets the time step counter.

        Args:
            seed (int | None): Seed for the environment. Defaults to None.
            options (dict[str, Any] | None): Additional options for the environment. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: observation and additional information.
        """
        state, info = super().reset(seed=seed, options=options)
        self.has_reset = True
        self.t = 0
        self.tunable_params = copy.deepcopy(self.init_initial_params)

        delta_change = {param_name: 0 for param_name in self.tunable_params.keys()}
        env_change = {param_name: 0 for param_name in self.tunable_params.keys()}

        obs = {
            "state": state,
            "env_change": env_change,
            "delta_change": delta_change,
            "relative_time": self.t,
        }

        info["Ground Truth Env Change"] = env_change
        info["Ground Truth Delta Change"] = delta_change

        return obs, info

    def freeze(self, mode: bool = True):
        """ "Freezes" the current MDP so that the environment dynamics do not change.
        Args:
            mode (bool): Boolean flag indicating whether to freeze the environment or not. Defaults to True.
        """
        if not isinstance(mode, bool):
            raise TypeError(f"Expected mode to be a boolean, got {type(mode)}")
        self.frozen = mode
        return self

    def unfreeze(self):
        """Unfreeze the environment dynamics for simulation.

        This function "unfreezes" the current MDP so that the environment dynamics can change.
        """
        return self.freeze(False)

    def __deepcopy__(self, memo):
        """Keeps track of deepcopying for the environment.

        If a derived class of this environement is made we set a flag to indicate that the environment is the simulation environment.

        This is the intended behavior for the deepcopy function.
        ```python
        env = gym.make("FrozenLake-v1")
        env = NSFrozenLakeWrapper(env,updatefn,is_slippery=False)
        sim_env = deepcopy(env)
        ```
        Then `sim_env.is_sim_env` will be set to True.

        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_planning_env(self):
        """Get the planning environment.

        Returns a copy of the current environment in its current state but the "transition function" is set to the initial transition function. Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_default_params(self):
        """Get dictionary of default parameters and their initial values

        Returns:
            dict[str,SupportsFloat]: Dictionary of parameter names and their initial values.
        """
        try:
            return TUNABLE_PARAMS[self.unwrapped.__class__.__name__]
        except KeyError:
            raise NotImplementedError(
                f"Default parameters for {self.unwrapped.__class__.__name__} not included in TUNABLE_PARAMS. Please add them to the dictionary in the base.py file."
            )

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        """Change the string representation of the environment so that user can see what/how parameters are being updated."""
        return super().__str__()


class Agent(ABC):
    """Base class for agents."""

    def __init__(self) -> None:
        super().__init__()

    def act(self, obs, *args, **kwargs) -> Any:
        """Agent decision making function. Subclasses must implement this method.

        Args:
            obs: Observation from the environment

        Returns:
            Any: Action to be taken by the agent
        """
        raise NotImplementedError("Subclasses must implement this method")


class StableBaselineWrapper:
    """Interface for StableBaseline3 Models and NS-Gym environments.
    Makes it so that you can call the stable baseline functions as you would other NS_Gym agents.
    """

    def __init__(self, model):
        """
        Args:
            model (Any): StableBaseline3 model
        """
        self.model = model

    def act(self, obs, *args, **kwargs) -> Any:
        """Agent decision making function. Calls the predict function of the StableBaseline3 model.
        Args:
            obs: Observation from the environment
        """
        action, _states = self.model.predict(obs)
        return action


class Evaluator(ABC):
    """Evaluator base class. This class is used to evaluate the difficulty of a transition between two environments."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def evaluate(self, env_1: Type[Env], env_2: Type[Env], *args, **kwargs) -> float:
        """Evaluate the difficulty of transitioning from env_1 to env_2. Subclasses must implement this method.

        Args:
            env_1 (Type[Env]): The initial environment
            env_2 (Type[Env]): The target environment
        """
        raise NotImplementedError("Subclas`ses must implement this method")

    def __call__(self):
        return self.evaluate()


def _generate_tunable_params():
    """
    Dynamically generates the tunable parameters dictionary by inspecting
    live Gymnasium environments for their default parameter values.
    """
    # This map defines which attributes to extract from non-MuJoCo environments.
    ATTRIBUTE_MAP = {
        "FrozenLakeEnv": ["P"],
        "CliffWalkingEnv": ["P"],
        "CartPoleEnv": [
            "gravity",
            "masscart",
            "masspole",
            "force_mag",
            "tau",
            "length",
        ],
        "AcrobotEnv": [
            "dt",
            "LINK_LENGTH_1",
            "LINK_LENGTH_2",
            "LINK_MASS_1",
            "LINK_MASS_2",
            "LINK_COM_POS_1",
            "LINK_COM_POS_2",
            "LINK_MOI",
        ],
        "MountainCarEnv": ["gravity", "force"],
        "Continuous_MountainCarEnv": ["power"],
        "PendulumEnv": ["m", "l", "dt", "g"],
    }

    env_ids_to_inspect = [
        "FrozenLake-v1",
        "CliffWalking-v1",
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "Ant-v5",
        "HalfCheetah-v5",
        "Hopper-v5",
        "Humanoid-v5",
        "HumanoidStandup-v5",
        "InvertedDoublePendulum-v5",
        "InvertedPendulum-v5",
        "Pusher-v5",
        "Reacher-v5",
        "Swimmer-v5",
    ]

    generated_params = {}

    for env_id in env_ids_to_inspect:
        try:
            env = gym.make(env_id)
            unwrapped_env = env.unwrapped
            env_name = unwrapped_env.__class__.__name__
            params = {}

            if env_name in ATTRIBUTE_MAP:
                for attr in ATTRIBUTE_MAP[env_name]:
                    if hasattr(unwrapped_env, attr):
                        params[attr] = getattr(unwrapped_env, attr)
            elif env_name in MUJOCO_GETTERS:
                for name, getter in MUJOCO_GETTERS[env_name].items():
                    value = getter[0](unwrapped_env)
                    # Use copy() for numpy arrays to prevent shared references
                    params[name] = (
                        np.copy(value) if isinstance(value, np.ndarray) else value
                    )

            if params:
                generated_params[env_name] = params
            env.close()
        except Exception as e:
            raise RuntimeError(f"Could not process {env_id}: {e}")

    return generated_params


MUJOCO_GETTERS = {
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
        #     # Note: Corrected 'floor' to 'ground' to match the XML file
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
            lambda env: env.model.body_mass[env.model.body("r_shoulder_pan_link").id],
            lambda env, val: env.model.body_mass.__setitem__(
                env.model.body("r_shoulder_pan_link").id, val
            ),
        ),
        "r_shoulder_lift_link_mass": (
            lambda env: env.model.body_mass[env.model.body("r_shoulder_lift_link").id],
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
            lambda env: env.model.dof_damping[env.model.joint("r_elbow_flex_joint").id],
            lambda env, val: env.model.dof_damping.__setitem__(
                env.model.joint("r_elbow_flex_joint").id, val
            ),
        ),
    },
}

MUJOCO_GETTERS["HumanoidStandupEnv"] = MUJOCO_GETTERS["HumanoidEnv"]


SUPPORTED_MUJOCO_ENV_IDS = [
    "Ant-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Reacher-v5",
    "Swimmer-v5",
    "Pusher-v5",
]


SUPPORTED_CLASSIC_CONTROL_ENV_IDS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
]


SUPPORTED_GRID_WORLD_ENV_IDS = [
    "CliffWalking-v1",
    "FrozenLake-v1",
    # "ns_gym/Bridge-v0" 
]

"""
Tunable parameters dictionary. Keys are environment names and values are dictionaries of parameter names and their initial values.
"""
TUNABLE_PARAMS = _generate_tunable_params()


if __name__ == "__main__":
    pass
