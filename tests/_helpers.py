"""Shared constants, the environment registry and helpers for the ns_gym tests.

Three things live here so that test modules do not each re-declare them:

* **Env id lists**, taken from ``ns_gym.base`` so they cannot drift from the package.
* **The env registry** (``ALL_CASES``): one :class:`EnvCase` per supported
  environment. A case knows how to build a wrapped env with one tunable
  parameter, an update sized to that parameter, and how to read the value the
  *base* env actually uses. Tests that should hold for every environment
  parametrize over the registry instead of naming envs one by one.
* **Known-bug markers** (:func:`known_bug`): a strict ``xfail`` carrying the id
  of a finding from "ns_gym: bug findings and fix plan". A test marked this way
  documents a confirmed bug. When the bug is fixed the test starts passing, the
  strict flag turns that into a failure, and the marker has to be removed in the
  same change as the fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import register, registry

from ns_gym import base
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import (
    DistributionDecrementUpdate,
    DistributionNoUpdate,
    IncrementUpdate,
    NoUpdate,
)
from ns_gym.wrappers import (
    MujocoWrapper,
    NSBridgeWrapper,
    NSClassicControlWrapper,
    NSCliffWalkingWrapper,
    NSFrozenLakeWrapper,
)
from ns_gym.wrappers.mujoco_env import param_look_up

# --------------------------------------------------------------------------
# Env ids
# --------------------------------------------------------------------------

CLASSIC_CONTROL_ENV_IDS = list(base.SUPPORTED_CLASSIC_CONTROL_ENV_IDS)
GRIDWORLD_ENV_IDS = list(base.SUPPORTED_GRID_WORLD_ENV_IDS)
MUJOCO_ENV_IDS = list(base.SUPPORTED_MUJOCO_ENV_IDS)
BRIDGE_ENV_ID = "ns_gym/Bridge-v0"

OBS_KEYS = ["state", "env_change", "delta_change", "relative_time"]

GRIDWORLD_WRAPPERS = {
    "CliffWalking-v1": NSCliffWalkingWrapper,
    "FrozenLake-v1": NSFrozenLakeWrapper,
}

#: Class name of the unwrapped env for each MuJoCo id. ``base.TUNABLE_PARAMS``
#: and ``param_look_up`` are keyed by class name; having the mapping here lets
#: tests be parametrized by (env, parameter) at collection time.
MUJOCO_ENV_CLASS = {
    "Ant-v5": "AntEnv",
    "HalfCheetah-v5": "HalfCheetahEnv",
    "Hopper-v5": "HopperEnv",
    "Humanoid-v5": "HumanoidEnv",
    "HumanoidStandup-v5": "HumanoidStandupEnv",
    "InvertedPendulum-v5": "InvertedPendulumEnv",
    "InvertedDoublePendulum-v5": "InvertedDoublePendulumEnv",
    "Reacher-v5": "ReacherEnv",
    "Swimmer-v5": "SwimmerEnv",
    "Pusher-v5": "PusherEnv",
}

#: Every (env id, tunable parameter) pair the MuJoCo wrapper accepts today.
MUJOCO_ENV_PARAMS = [
    (env_id, param)
    for env_id in MUJOCO_ENV_IDS
    for param in base.TUNABLE_PARAMS[MUJOCO_ENV_CLASS[env_id]]
]

# --------------------------------------------------------------------------
# Known bugs
# --------------------------------------------------------------------------

#: Finding id -> one-line description. Ids match the findings document; the
#: three that are also GitHub issues say so.
KNOWN_BUGS = {
    "C1": "CartPole planning env / reset leaves total_mass and polemass_length inconsistent with masspole",
    "C2": "planning envs are rebuilt from a hard-coded id and drop the base env's constructor arguments",
    "C3": "CliffWalking planning env loses modified_rewards and terminal_cliff",
    "D2": "docs list friction parameters that have no physical effect and that the wrapper rejects",
    "D5": "Bridge docs name the split parameters left_side/right_side; the wrapper accepts P_left/P_right",
    "E1": "ExponentialDecay compounds exp(-lambda*t) on every call instead of Y_0*exp(-lambda*t)",
    "E3": "BudgetBoundedIncrement cannot be constructed",
    "E4": "DistributionIncrementUpdate/DistributionDecrementUpdate clamp one side only and leave [0, 1]",
    "E5": "StepWiseUpdate reports a change after its list is exhausted and empties the caller's list",
    "F1": "MCTS backpropagation discounts by gamma**depth cumulatively",
    "F2": "MCTS.search raises KeyError when a root action was never visited",
    "F4": "ns_gym/VehicleTracking-v0 is registered but its module does not exist",
    "G1": "Acrobot LINK_LENGTH_2 may shrink below its centre-of-mass position",
    "G2": "freeze() hides notifications but parameters keep changing",
    "G3": "MountainCar emits a UserWarning on every deepcopy",
}


def known_bug(bug_id: str, raises: Any = AssertionError):
    """Strict ``xfail`` for a confirmed bug.

    ``raises`` pins the failure to the exception the bug produces, so a typo in
    the test (``NameError``, ``TypeError``...) is reported as a real failure
    instead of being counted as the expected one.
    """
    return pytest.mark.xfail(
        strict=True, raises=raises, reason=f"{bug_id}: {KNOWN_BUGS[bug_id]}"
    )


# --------------------------------------------------------------------------
# Env registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvCase:
    """One supported environment plus a representative tunable parameter."""

    env_id: str
    family: str  # "classic_control" | "gridworld" | "bridge" | "mujoco"
    wrapper: type
    cls_name: str
    param: str
    #: Initial slip distribution for the transition-probability envs.
    initial_dist: Optional[tuple] = None

    # -- construction -------------------------------------------------------

    @property
    def is_distribution(self) -> bool:
        return self.initial_dist is not None

    def initial(self):
        """Value of the parameter before any update."""
        if self.is_distribution:
            return [float(p) for p in self.initial_dist]
        return float(np.asarray(base.TUNABLE_PARAMS[self.cls_name][self.param]))

    def step_size(self) -> float:
        """Per-firing increment sized to the parameter (5% of its default)."""
        return 0.05 * abs(self.initial())

    def update_fn(self, scheduler=None):
        """A valid update for this parameter, firing on ``scheduler``."""
        scheduler = scheduler if scheduler is not None else ContinuousScheduler()
        if self.is_distribution:
            return DistributionDecrementUpdate(scheduler, k=0.1)
        return IncrementUpdate(scheduler, k=self.step_size())

    def no_update_fn(self):
        """An update that never changes the parameter."""
        if self.is_distribution:
            return DistributionNoUpdate(ContinuousScheduler())
        return NoUpdate(ContinuousScheduler())

    def make(self, update_fn=None, *, scheduler=None, make_kwargs=None, **wrapper_kwargs):
        """Build the wrapped env with ``{self.param: update_fn}``."""
        env = gym.make(self.env_id, **(make_kwargs or {}))
        fn = update_fn if update_fn is not None else self.update_fn(scheduler)
        if self.is_distribution:
            wrapper_kwargs.setdefault("initial_prob_dist", list(self.initial()))
        return self.wrapper(env, {self.param: fn}, **wrapper_kwargs)

    # -- observation of the parameter --------------------------------------

    def executed(self, env):
        """The value the *base* env will use on its next step."""
        u = env.unwrapped
        if self.family == "classic_control":
            return float(getattr(u, self.param))
        if self.family == "mujoco":
            getter, _ = param_look_up(self.cls_name, self.param)[0]
            return float(getter(u))
        if self.family == "bridge":
            return [float(p) for p in u.P]
        return executed_probs(env)

    def reported(self, env):
        """The value the wrapper reports to the agent."""
        if self.family == "gridworld":
            return [float(p) for p in env.transition_prob]
        return self.executed(env)


def executed_probs(env):
    """Outcome probabilities the base gridworld env samples from.

    Read from the transition table the env actually steps with, at the first
    non-terminal state (terminal states have a single outcome).
    """
    P = env.unwrapped.P
    s = next(s for s in sorted(P) if len(P[s][0]) > 1)
    return [float(t[0]) for t in P[s][0]]


def same(a, b) -> bool:
    """Numerical equality for scalars and probability lists."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-9, atol=1e-12))


_CC_PARAM = {
    "CartPole-v1": ("CartPoleEnv", "masspole"),
    "Acrobot-v1": ("AcrobotEnv", "LINK_MASS_2"),
    "MountainCar-v0": ("MountainCarEnv", "force"),
    "MountainCarContinuous-v0": ("Continuous_MountainCarEnv", "power"),
    "Pendulum-v1": ("PendulumEnv", "m"),
}

_MUJOCO_PARAM = {
    "Ant-v5": "torso_mass",
    "HalfCheetah-v5": "torso_mass",
    "Hopper-v5": "torso_mass",
    "Humanoid-v5": "torso_mass",
    "HumanoidStandup-v5": "torso_mass",
    "InvertedPendulum-v5": "pole_mass",
    "InvertedDoublePendulum-v5": "cart_mass",
    "Reacher-v5": "body0_mass",
    "Swimmer-v5": "body_mid_mass",
    "Pusher-v5": "r_shoulder_pan_link_mass",
}

CLASSIC_CONTROL_CASES = [
    EnvCase(env_id, "classic_control", NSClassicControlWrapper, *_CC_PARAM[env_id])
    for env_id in CLASSIC_CONTROL_ENV_IDS
]

GRIDWORLD_CASES = [
    EnvCase("CliffWalking-v1", "gridworld", NSCliffWalkingWrapper, "CliffWalkingEnv", "P", (1.0, 0.0, 0.0, 0.0)),
    EnvCase("FrozenLake-v1", "gridworld", NSFrozenLakeWrapper, "FrozenLakeEnv", "P", (1.0, 0.0, 0.0)),
]

BRIDGE_CASES = [
    EnvCase(BRIDGE_ENV_ID, "bridge", NSBridgeWrapper, "Bridge", "P", (1.0, 0.0, 0.0)),
]

MUJOCO_CASES = [
    EnvCase(env_id, "mujoco", MujocoWrapper, MUJOCO_ENV_CLASS[env_id], _MUJOCO_PARAM[env_id])
    for env_id in MUJOCO_ENV_IDS
]

#: Envs whose non-stationary parameter is a transition-probability distribution.
DISTRIBUTION_CASES = GRIDWORLD_CASES + BRIDGE_CASES

ALL_CASES = CLASSIC_CONTROL_CASES + GRIDWORLD_CASES + BRIDGE_CASES + MUJOCO_CASES

CASE_BY_ID = {case.env_id: case for case in ALL_CASES}


def env_cases(cases=None, known_bugs=None):
    """``pytest.param`` list for ``parametrize("case", ...)``.

    ``known_bugs`` maps env id -> finding id for the envs on which the test is
    expected to fail today.
    """
    cases = ALL_CASES if cases is None else cases
    known_bugs = known_bugs or {}
    return [
        pytest.param(
            case,
            id=case.env_id,
            marks=[known_bug(known_bugs[case.env_id])] if case.env_id in known_bugs else [],
        )
        for case in cases
    ]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def validate_no_double_wrap(env, wrapper_class):
    """``env`` is a ``wrapper_class`` and that class appears nowhere below it."""
    assert isinstance(env, wrapper_class), (
        f"Expected outermost wrapper to be {wrapper_class.__name__}, got {type(env).__name__}"
    )
    current = env.env
    depth = 1
    while hasattr(current, "env"):
        assert not isinstance(current, wrapper_class), (
            f"Double wrap detected: {wrapper_class.__name__} again at depth {depth}"
        )
        current = current.env
        depth += 1
    assert not isinstance(current, wrapper_class), (
        f"Double wrap detected: {wrapper_class.__name__} is the base environment"
    )


def fixed_actions(env, n: int, seed: int = 0):
    """``n`` reproducible actions for ``env`` that do not touch its own RNG."""
    space = env.action_space
    if isinstance(space, gym.spaces.Discrete):
        rng = np.random.default_rng(seed)
        return [int(a) for a in rng.integers(0, space.n, size=n)]
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(space.low, space.high).astype(space.dtype) for _ in range(n)
    ]


def state_of(obs):
    """The raw state from an ns_gym observation dict."""
    return obs["state"] if isinstance(obs, dict) else obs


def register_ns_env(custom_id: str, make_wrapped):
    """Register ``make_wrapped(**kwargs)`` under ``custom_id`` the way ns_gym
    documents it: no env checker and no order-enforcing wrapper on top."""
    if custom_id in registry:
        del registry[custom_id]
    register(
        id=custom_id,
        entry_point=make_wrapped,
        disable_env_checker=True,
        order_enforce=False,
    )
    return custom_id


def unregister(custom_id: str):
    if custom_id in registry:
        del registry[custom_id]


# --------------------------------------------------------------------------
# Smaller env subsets used by the seeding and reset tests
# --------------------------------------------------------------------------

#: Two cheap MuJoCo envs, enough to exercise the wrapper where running all ten
#: would only repeat the same code path.
MUJOCO_TIME_TEST_IDS = [
    "InvertedPendulum-v5",
    "HalfCheetah-v5",
]

#: The parameter each classic-control env random-walks in the seeding tests.
CC_SEEDING_PARAMS = {
    "CartPole-v1": "masspole",
    "Acrobot-v1": "LINK_LENGTH_1",
    "MountainCar-v0": "force",
    "MountainCarContinuous-v0": "power",
    "Pendulum-v1": "m",
}


# --------------------------------------------------------------------------
# Action sequences that keep an episode alive
# --------------------------------------------------------------------------

#: Under deterministic dynamics these never reach a terminal cell before their
#: last step, so tests can rely on an episode lasting as long as the list.
_SAFE_GRID_ACTIONS = {
    # FrozenLake 4x4: 0 -> 4 -> 8 -> 9 -> 13 -> 14 -> 15 (goal)
    "FrozenLake-v1": [1, 1, 2, 1, 2, 2],
    # CliffWalking: up the left edge, then right along the top rows
    "CliffWalking-v1": [0, 0, 0, 1, 1, 1],
    # Bridge: step right and back along the bridge row
    BRIDGE_ENV_ID: [2, 0, 2, 0, 2, 0],
}


def safe_actions(case: EnvCase, env, n: int):
    """``n`` reproducible actions that do not end the episode early.

    The grid envs get a fixed path (at most six steps); the continuous-state
    envs get seeded random actions, which cannot terminate them this quickly.
    """
    if case.env_id in _SAFE_GRID_ACTIONS:
        path = _SAFE_GRID_ACTIONS[case.env_id]
        assert n <= len(path), f"safe path for {case.env_id} has only {len(path)} steps"
        return path[:n]
    return fixed_actions(env, n)
