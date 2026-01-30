import pytest
import gymnasium as gym
import numpy as np
from copy import deepcopy

import ns_gym.base as base
from ns_gym.base import TUNABLE_PARAMS, Reward
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate, DistributionIncrementUpdate
from ns_gym.wrappers import NSClassicControlWrapper, NSCliffWalkingWrapper, NSFrozenLakeWrapper


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

OBS_KEYS = ["state", "env_change", "delta_change", "relative_time"]


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
def gw_wrappers():
    return {
        "CliffWalking-v1": NSCliffWalkingWrapper,
        "FrozenLake-v1": NSFrozenLakeWrapper,
    }


@pytest.fixture
def gw_params():
    fn = DistributionIncrementUpdate(ContinuousScheduler(), k=-0.1)
    return {
        "CliffWalking-v1": {"P": fn},
        "FrozenLake-v1": {"P": fn},
    }


# ============================================================
# Reset: restores all params to initial values
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_reset_restores_all_params_classic_control(cc_params, env_id):
    env = gym.make(env_id)
    ns_env = NSClassicControlWrapper(env, cc_params[env_id])
    ns_env.reset(seed=42)

    env_name = ns_env.unwrapped.__class__.__name__
    initial_params = {
        p: deepcopy(getattr(ns_env.unwrapped, p))
        for p in TUNABLE_PARAMS[env_name]
    }

    # Step enough times to change params
    for _ in range(10):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            break

    # Reset and verify all params restored
    ns_env.reset(seed=42)
    for p in TUNABLE_PARAMS[env_name]:
        assert np.isclose(getattr(ns_env.unwrapped, p), initial_params[p]), (
            f"Param '{p}' not restored after reset: "
            f"expected={initial_params[p]}, got={getattr(ns_env.unwrapped, p)}"
        )


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_reset_restores_all_params_gridworld(gw_wrappers, gw_params, env_id):
    env = gym.make(env_id)
    WrapperClass = gw_wrappers[env_id]
    ns_env = WrapperClass(env, gw_params[env_id])
    ns_env.reset(seed=42)

    initial_tp = ns_env.transition_prob[:]

    # Step to change params
    for _ in range(5):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            break

    ns_env.reset(seed=42)
    assert ns_env.transition_prob == initial_tp, (
        "transition_prob not restored after reset"
    )


# ============================================================
# Multiple resets
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_multiple_resets_classic_control(cc_params, env_id):
    env = gym.make(env_id)
    ns_env = NSClassicControlWrapper(env, cc_params[env_id])

    env_name = ns_env.unwrapped.__class__.__name__

    for cycle in range(3):
        obs, info = ns_env.reset(seed=42)

        # After reset: t should be 0, obs structure correct
        assert ns_env.t == 0, f"t != 0 after reset cycle {cycle}"
        assert isinstance(obs, dict)
        assert all(k in obs for k in OBS_KEYS)
        assert obs["relative_time"] == 0

        # All change notifications should be zero
        for p in cc_params[env_id]:
            assert obs["env_change"][p] == 0
            assert obs["delta_change"][p] == 0.0

        # All params should be at initial values
        for p in TUNABLE_PARAMS[env_name]:
            expected = TUNABLE_PARAMS[env_name][p]
            actual = getattr(ns_env.unwrapped, p)
            assert np.isclose(actual, expected), (
                f"Cycle {cycle}: param '{p}' = {actual}, expected {expected}"
            )

        # Step a few times
        for _ in range(5):
            obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
            if done or trunc:
                break


# ============================================================
# Step: t increments correctly
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_step_increments_t_classic_control(cc_params, env_id):
    env = gym.make(env_id)
    ns_env = NSClassicControlWrapper(env, cc_params[env_id])
    ns_env.reset(seed=42)

    for step_num in range(1, 6):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            break
        assert ns_env.t == step_num, f"Expected t={step_num}, got t={ns_env.t}"
        assert obs["relative_time"] == step_num


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_step_increments_t_gridworld(gw_wrappers, gw_params, env_id):
    env = gym.make(env_id)
    WrapperClass = gw_wrappers[env_id]
    ns_env = WrapperClass(env, gw_params[env_id])
    ns_env.reset(seed=42)

    for step_num in range(1, 6):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            break
        assert obs["relative_time"] == step_num


# ============================================================
# Step: updates params by known amount
# ============================================================

def test_step_updates_params_by_known_amount():
    """CartPole-specific: verify exact param update and dependency resolver."""
    env = gym.make("CartPole-v1")
    k = 0.5
    fn = IncrementUpdate(ContinuousScheduler(start=0), k=k)
    ns_env = NSClassicControlWrapper(env, {"masspole": fn})
    ns_env.reset(seed=42)

    initial_masspole = env.unwrapped.masspole
    initial_masscart = env.unwrapped.masscart
    initial_length = env.unwrapped.length

    ns_env.step(0)

    expected_masspole = initial_masspole + k
    assert np.isclose(env.unwrapped.masspole, expected_masspole), (
        f"masspole: expected {expected_masspole}, got {env.unwrapped.masspole}"
    )

    # Dependency resolver should update total_mass and polemass_length
    expected_total_mass = expected_masspole + initial_masscart
    expected_polemass_length = expected_masspole * initial_length
    assert np.isclose(env.unwrapped.total_mass, expected_total_mass)
    assert np.isclose(env.unwrapped.polemass_length, expected_polemass_length)


# ============================================================
# Step: notification flags
# ============================================================

def test_step_notification_false_false(cc_params):
    """change_notification=False, delta_change_notification=False: all zeros."""
    env = gym.make("CartPole-v1")
    ns_env = NSClassicControlWrapper(
        env, cc_params["CartPole-v1"],
        change_notification=False,
        delta_change_notification=False,
    )
    ns_env.reset(seed=42)
    obs, _, _, _, _ = ns_env.step(0)

    for p in cc_params["CartPole-v1"]:
        assert obs["env_change"][p] == 0
        assert obs["delta_change"][p] == 0.0


def test_step_notification_true_false(cc_params):
    """change_notification=True, delta_change_notification=False: env_change reflects, delta zeros."""
    env = gym.make("CartPole-v1")
    ns_env = NSClassicControlWrapper(
        env, cc_params["CartPole-v1"],
        change_notification=True,
        delta_change_notification=False,
    )
    ns_env.reset(seed=42)
    obs, _, _, _, _ = ns_env.step(0)

    # env_change should be nonzero (params updated with ContinuousScheduler)
    any_flagged = any(obs["env_change"][p] for p in cc_params["CartPole-v1"])
    assert any_flagged, "env_change should reflect actual changes"

    # delta_change should still be zeros
    for p in cc_params["CartPole-v1"]:
        assert obs["delta_change"][p] == 0.0


def test_step_notification_true_true(cc_params):
    """change_notification=True, delta_change_notification=True: both reflect."""
    env = gym.make("CartPole-v1")
    ns_env = NSClassicControlWrapper(
        env, cc_params["CartPole-v1"],
        change_notification=True,
        delta_change_notification=True,
    )
    ns_env.reset(seed=42)
    obs, _, _, _, _ = ns_env.step(0)

    any_flagged = any(obs["env_change"][p] for p in cc_params["CartPole-v1"])
    assert any_flagged, "env_change should reflect actual changes"

    any_delta = any(obs["delta_change"][p] != 0.0 for p in cc_params["CartPole-v1"])
    assert any_delta, "delta_change should reflect actual deltas"


def test_step_notification_false_true_raises():
    """delta_change_notification=True without change_notification=True should raise."""
    env = gym.make("CartPole-v1")
    fn = IncrementUpdate(ContinuousScheduler(), k=0.1)
    with pytest.raises(AssertionError):
        NSClassicControlWrapper(
            env, {"masspole": fn},
            change_notification=False,
            delta_change_notification=True,
        )


# ============================================================
# Step: scalar reward modes
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_step_scalar_reward_default(cc_params, env_id):
    env = gym.make(env_id)
    ns_env = NSClassicControlWrapper(env, cc_params[env_id], scalar_reward=True)
    ns_env.reset(seed=42)
    _, reward, _, _, _ = ns_env.step(ns_env.action_space.sample())
    assert isinstance(reward, (int, float, np.floating)), (
        f"Expected scalar reward, got {type(reward)}"
    )


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_step_reward_dataclass(cc_params, env_id):
    env = gym.make(env_id)
    ns_env = NSClassicControlWrapper(env, cc_params[env_id], scalar_reward=False)
    ns_env.reset(seed=42)
    _, reward, _, _, _ = ns_env.step(ns_env.action_space.sample())
    assert isinstance(reward, Reward), (
        f"Expected Reward dataclass, got {type(reward)}"
    )
    assert hasattr(reward, "reward")
    assert hasattr(reward, "env_change")
    assert hasattr(reward, "delta_change")
    assert hasattr(reward, "relative_time")


# ============================================================
# Step: observation structure
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_step_obs_structure_classic_control(cc_params, env_id):
    env = gym.make(env_id)
    ns_env = NSClassicControlWrapper(env, cc_params[env_id])

    obs, info = ns_env.reset(seed=42)
    assert isinstance(obs, dict)
    assert all(k in obs for k in OBS_KEYS), f"Missing keys in reset obs: {obs.keys()}"
    assert obs["relative_time"] == 0

    obs, _, _, _, _ = ns_env.step(ns_env.action_space.sample())
    assert isinstance(obs, dict)
    assert all(k in obs for k in OBS_KEYS), f"Missing keys in step obs: {obs.keys()}"
    assert obs["relative_time"] > 0


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_step_obs_structure_gridworld(gw_wrappers, gw_params, env_id):
    env = gym.make(env_id)
    WrapperClass = gw_wrappers[env_id]
    ns_env = WrapperClass(env, gw_params[env_id])

    obs, info = ns_env.reset(seed=42)
    assert isinstance(obs, dict)
    assert all(k in obs for k in OBS_KEYS)
    assert obs["relative_time"] == 0

    obs, _, _, _, _ = ns_env.step(ns_env.action_space.sample())
    assert isinstance(obs, dict)
    assert all(k in obs for k in OBS_KEYS)


# ============================================================
# Info: ground truth always present
# ============================================================

def test_info_contains_ground_truth(cc_params):
    """Ground truth env/delta change should be in info regardless of notification flags."""
    env = gym.make("CartPole-v1")
    ns_env = NSClassicControlWrapper(
        env, cc_params["CartPole-v1"],
        change_notification=False,
        delta_change_notification=False,
    )
    ns_env.reset(seed=42)
    _, _, _, _, info = ns_env.step(0)

    assert "Ground Truth Env Change" in info
    assert "Ground Truth Delta Change" in info
    assert isinstance(info["Ground Truth Env Change"], dict)
    assert isinstance(info["Ground Truth Delta Change"], dict)


# ============================================================
# Constraint checker
# ============================================================

def test_constraint_checker_prevents_invalid_values():
    """Large negative increment on masscart should be blocked by constraint checker."""
    from ns_gym.wrappers.classic_control import ConstraintViolationWarning

    env = gym.make("CartPole-v1")
    # k=-100 would drive masscart negative on the first step
    fn = IncrementUpdate(ContinuousScheduler(start=0), k=-100.0)
    ns_env = NSClassicControlWrapper(env, {"masscart": fn})
    ns_env.reset(seed=42)

    initial_masscart = env.unwrapped.masscart

    with pytest.warns(ConstraintViolationWarning):
        ns_env.step(0)

    # masscart should NOT have gone negative
    assert env.unwrapped.masscart > 0, (
        f"Constraint checker failed: masscart={env.unwrapped.masscart}"
    )
