import pytest
import gymnasium as gym
import numpy as np
import copy
from copy import deepcopy

from gymnasium.envs.registration import register, registry

from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate, DistributionIncrementUpdate
from ns_gym.wrappers import NSClassicControlWrapper, NSCliffWalkingWrapper, NSFrozenLakeWrapper
from ns_gym.base import TUNABLE_PARAMS, Reward


# --- Constants ---

CLASSIC_CONTROL_ENV_IDS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
]

GRIDWORLD_ENV_IDS = [
    "CliffWalking-v1",
    "FrozenLake-v1",
]

GRIDWORLD_WRAPPER_MAP = {
    "CliffWalking-v1": NSCliffWalkingWrapper,
    "FrozenLake-v1": NSFrozenLakeWrapper,
}


# --- Fixtures ---

@pytest.fixture
def cc_params():
    fn = IncrementUpdate(ContinuousScheduler(), k=0.1)
    dec_fn = IncrementUpdate(ContinuousScheduler(), k=-0.1)
    return {
        "CartPole-v1": {"masspole": fn, "gravity": fn},
        "Acrobot-v1": {"LINK_LENGTH_1": fn, "LINK_MASS_2": fn},
        "MountainCar-v0": {"gravity": dec_fn, "force": fn},
        "MountainCarContinuous-v0": {"power": fn},
        "Pendulum-v1": {"m": fn, "g": fn},
    }


@pytest.fixture
def gw_params():
    fn = DistributionIncrementUpdate(ContinuousScheduler(), k=-0.1)
    return {
        "CliffWalking-v1": {"P": fn},
        "FrozenLake-v1": {"P": fn},
    }


# --- Helpers ---

def _validate_no_double_wrap(env, wrapper_class):
    """Assert outermost is wrapper_class and no double wrapping."""
    assert isinstance(env, wrapper_class), (
        f"Expected {wrapper_class.__name__}, got {type(env).__name__}"
    )
    current = env.env
    while hasattr(current, "env"):
        assert not isinstance(current, wrapper_class), "Double wrap detected"
        current = current.env
    assert not isinstance(current, wrapper_class)


def _register_cc_env(env_id, tunable_params, custom_id, **wrapper_kwargs):
    """Register a classic control NS env and return the custom_id."""
    def _make_custom_env(**kwargs):
        # Filter out gymnasium's own kwargs that shouldn't go to the wrapper
        base_env = gym.make(env_id, **kwargs)
        return NSClassicControlWrapper(
            base_env, tunable_params, **wrapper_kwargs
        )

    if custom_id in registry:
        del registry[custom_id]

    register(
        id=custom_id,
        entry_point=_make_custom_env,
        disable_env_checker=True,
        order_enforce=False,
    )
    return custom_id


def _register_gw_env(env_id, tunable_params, custom_id, **wrapper_kwargs):
    """Register a gridworld NS env and return the custom_id."""
    WrapperClass = GRIDWORLD_WRAPPER_MAP[env_id]

    def _make_custom_env(**kwargs):
        base_env = gym.make(env_id, **kwargs)
        return WrapperClass(base_env, tunable_params, **wrapper_kwargs)

    if custom_id in registry:
        del registry[custom_id]

    register(
        id=custom_id,
        entry_point=_make_custom_env,
        disable_env_checker=True,
        order_enforce=False,
    )
    return custom_id


def _cleanup_registry(custom_id):
    if custom_id in registry:
        del registry[custom_id]


# ============================================================
# Registration: wrapper stack integrity
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_registered_env_wrapper_stack_classic_control(cc_params, env_id):
    custom_id = f"TestReg-CC-{env_id}"
    try:
        _register_cc_env(env_id, cc_params[env_id], custom_id)
        ns_env = gym.make(custom_id)
        _validate_no_double_wrap(ns_env, NSClassicControlWrapper)
    finally:
        _cleanup_registry(custom_id)


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_registered_env_wrapper_stack_gridworld(gw_params, env_id):
    WrapperClass = GRIDWORLD_WRAPPER_MAP[env_id]
    custom_id = f"TestReg-GW-{env_id}"
    try:
        _register_gw_env(env_id, gw_params[env_id], custom_id)
        ns_env = gym.make(custom_id)
        _validate_no_double_wrap(ns_env, WrapperClass)
    finally:
        _cleanup_registry(custom_id)


# ============================================================
# Registration: deepcopy works after gym.make
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_registered_env_deepcopy_works_classic_control(cc_params, env_id):
    if env_id in ["MountainCar-v0", "MountainCarContinuous-v0"]:
        pytest.skip("Known deepcopy state divergence issue for MountainCar")

    custom_id = f"TestRegDC-CC-{env_id}"
    try:
        _register_cc_env(env_id, cc_params[env_id], custom_id)
        ns_env = gym.make(custom_id)
        ns_env.reset(seed=42)

        for _ in range(3):
            ns_env.step(ns_env.action_space.sample())

        # This is the critical test: deepcopy should NOT raise TypeError
        # from kwargs leaking into the base env constructor
        sim_env = copy.deepcopy(ns_env)

        assert sim_env.is_sim_env is True
        assert isinstance(sim_env, NSClassicControlWrapper)
        _validate_no_double_wrap(sim_env, NSClassicControlWrapper)
    finally:
        _cleanup_registry(custom_id)


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_registered_env_deepcopy_works_gridworld(gw_params, env_id):
    WrapperClass = GRIDWORLD_WRAPPER_MAP[env_id]
    custom_id = f"TestRegDC-GW-{env_id}"
    try:
        _register_gw_env(env_id, gw_params[env_id], custom_id)
        ns_env = gym.make(custom_id)
        ns_env.reset(seed=42)
        ns_env.step(ns_env.action_space.sample())

        sim_env = copy.deepcopy(ns_env)

        assert sim_env.is_sim_env is True
        assert isinstance(sim_env, WrapperClass)
    finally:
        _cleanup_registry(custom_id)


# ============================================================
# Registration: get_planning_env works after gym.make
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_registered_env_get_planning_env_classic_control(cc_params, env_id):
    custom_id = f"TestRegPE-CC-{env_id}"
    try:
        _register_cc_env(env_id, cc_params[env_id], custom_id)
        ns_env = gym.make(custom_id)
        ns_env.reset(seed=42)
        ns_env.step(ns_env.action_space.sample())

        planning_env = ns_env.get_planning_env()

        assert planning_env.is_sim_env is True
        assert isinstance(planning_env, NSClassicControlWrapper)
        _validate_no_double_wrap(planning_env, NSClassicControlWrapper)
    finally:
        _cleanup_registry(custom_id)


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_registered_env_get_planning_env_gridworld(gw_params, env_id):
    WrapperClass = GRIDWORLD_WRAPPER_MAP[env_id]
    custom_id = f"TestRegPE-GW-{env_id}"
    try:
        _register_gw_env(env_id, gw_params[env_id], custom_id)
        ns_env = gym.make(custom_id)
        ns_env.reset(seed=42)
        ns_env.step(ns_env.action_space.sample())

        planning_env = ns_env.get_planning_env()

        assert planning_env.is_sim_env is True
        assert isinstance(planning_env, WrapperClass)
        _validate_no_double_wrap(planning_env, WrapperClass)
    finally:
        _cleanup_registry(custom_id)


# ============================================================
# Registration: kwargs leak prevention (the critical bug test)
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_registration_with_notifications_no_kwargs_leak(cc_params, env_id):
    """Register with change_notification=True. Deepcopy must not leak these
    kwargs into the base env constructor (CartPoleEnv doesn't accept them)."""
    if env_id in ["MountainCar-v0", "MountainCarContinuous-v0"]:
        pytest.skip("Known deepcopy state divergence issue for MountainCar")

    custom_id = f"TestRegKW-CC-{env_id}"
    try:
        _register_cc_env(
            env_id, cc_params[env_id], custom_id,
            change_notification=True,
            delta_change_notification=True,
        )
        ns_env = gym.make(custom_id)
        ns_env.reset(seed=42)
        ns_env.step(ns_env.action_space.sample())

        # This would raise TypeError if kwargs leak:
        # "CartPoleEnv.__init__() got an unexpected keyword argument 'change_notification'"
        sim_env = copy.deepcopy(ns_env)

        assert sim_env.is_sim_env is True
        assert isinstance(sim_env, NSClassicControlWrapper)

        # Verify the copy is functional
        obs, _, _, _, _ = sim_env.step(sim_env.action_space.sample())
        assert isinstance(obs, dict)
        assert all(k in obs for k in ["state", "env_change", "delta_change", "relative_time"])
    finally:
        _cleanup_registry(custom_id)


# ============================================================
# Registration: scalar_reward propagates through deepcopy
# ============================================================

@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_registration_with_scalar_reward_propagates(cc_params, env_id):
    custom_id = f"TestRegSR-CC-{env_id}"
    try:
        _register_cc_env(
            env_id, cc_params[env_id], custom_id,
            scalar_reward=False,
        )
        ns_env = gym.make(custom_id)
        ns_env.reset(seed=42)

        _, reward, _, _, _ = ns_env.step(ns_env.action_space.sample())
        assert isinstance(reward, Reward), (
            f"Expected Reward dataclass from registered env, got {type(reward)}"
        )

        # Deepcopy should preserve scalar_reward=False
        sim_env = copy.deepcopy(ns_env)
        assert sim_env.scalar_reward is False

        _, sim_reward, _, _, _ = sim_env.step(sim_env.action_space.sample())
        assert isinstance(sim_reward, Reward), (
            f"Expected Reward dataclass from deepcopied env, got {type(sim_reward)}"
        )
    finally:
        _cleanup_registry(custom_id)


# ============================================================
# Registration: full episode then reset
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_registered_env_full_episode(cc_params, env_id):
    custom_id = f"TestRegEp-CC-{env_id}"
    try:
        _register_cc_env(env_id, cc_params[env_id], custom_id)
        ns_env = gym.make(custom_id)
        ns_env.reset(seed=42)

        # Run a short episode
        for _ in range(20):
            obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
            if done or trunc:
                break

        # Reset and verify clean state
        obs, info = ns_env.reset(seed=42)
        assert ns_env.t == 0
        assert obs["relative_time"] == 0
        assert isinstance(obs, dict)
        assert all(k in obs for k in ["state", "env_change", "delta_change", "relative_time"])
    finally:
        _cleanup_registry(custom_id)


# ============================================================
# Registration: step updates params (not frozen by default)
# ============================================================

def test_registered_env_step_updates_params():
    """Registered env step should update params (is_sim_env defaults to False)."""
    k = 0.5
    fn = IncrementUpdate(ContinuousScheduler(start=0), k=k)
    tunable_params = {"masspole": fn}
    custom_id = "TestRegStep-CartPole-v1"

    try:
        _register_cc_env("CartPole-v1", tunable_params, custom_id)
        ns_env = gym.make(custom_id)
        ns_env.reset(seed=42)

        initial_masspole = ns_env.unwrapped.masspole
        ns_env.step(0)

        expected = initial_masspole + k
        assert np.isclose(ns_env.unwrapped.masspole, expected), (
            f"Registered env param not updated: expected {expected}, "
            f"got {ns_env.unwrapped.masspole}"
        )
    finally:
        _cleanup_registry(custom_id)
