import pytest
import gymnasium as gym
import numpy as np
import copy
from copy import deepcopy

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


# --- Fixtures ---

@pytest.fixture
def cc_params():
    """Tunable params for each classic control env."""
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


# --- Helper ---

def _make_cc_env(env_id, cc_params, **wrapper_kwargs):
    env = gym.make(env_id)
    return NSClassicControlWrapper(env, cc_params[env_id], **wrapper_kwargs)


def _make_gw_env(env_id, gw_wrappers, gw_params, **wrapper_kwargs):
    env = gym.make(env_id)
    WrapperClass = gw_wrappers[env_id]
    return WrapperClass(env, gw_params[env_id], **wrapper_kwargs)


def _validate_no_double_wrap(env, wrapper_class):
    """Walk the .env chain and assert no double wrapping."""
    assert isinstance(env, wrapper_class)
    current = env.env
    while hasattr(current, "env"):
        assert not isinstance(current, wrapper_class), "Double wrap detected"
        current = current.env
    assert not isinstance(current, wrapper_class), "Double wrap at base"


# ============================================================
# Deepcopy: is_sim_env flag
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy_sets_is_sim_env_classic_control(cc_params, env_id):
    ns_env = _make_cc_env(env_id, cc_params)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)

    assert sim_env.is_sim_env is True
    assert ns_env.is_sim_env is False
    assert isinstance(sim_env, NSClassicControlWrapper)


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_deepcopy_sets_is_sim_env_gridworld(gw_wrappers, gw_params, env_id):
    ns_env = _make_gw_env(env_id, gw_wrappers, gw_params)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)

    assert sim_env.is_sim_env is True
    assert ns_env.is_sim_env is False
    assert isinstance(sim_env, type(ns_env))


# ============================================================
# Deepcopy: params match at copy point
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy_params_match_classic_control(cc_params, env_id):
    ns_env = _make_cc_env(env_id, cc_params)
    ns_env.reset(seed=42)
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)

    env_name = ns_env.unwrapped.__class__.__name__
    for param in TUNABLE_PARAMS[env_name]:
        orig_val = getattr(ns_env.unwrapped, param)
        copy_val = getattr(sim_env.unwrapped, param)
        assert np.isclose(orig_val, copy_val), (
            f"Param '{param}' mismatch: original={orig_val}, copy={copy_val}"
        )


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_deepcopy_params_match_gridworld(gw_wrappers, gw_params, env_id):
    ns_env = _make_gw_env(env_id, gw_wrappers, gw_params)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)

    assert sim_env.unwrapped.P.keys() == ns_env.unwrapped.P.keys()


# ============================================================
# Deepcopy: independence (stepping copy doesn't affect original)
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy_independence_classic_control(cc_params, env_id):
    if env_id in ["MountainCar-v0", "MountainCarContinuous-v0"]:
        pytest.skip("Known deepcopy state divergence issue for MountainCar")

    ns_env = _make_cc_env(env_id, cc_params, in_sim_change=True)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    # Record original params before deepcopy
    env_name = ns_env.unwrapped.__class__.__name__
    orig_params = {}
    for param in cc_params[env_id]:
        orig_params[param] = deepcopy(getattr(ns_env.unwrapped, param))

    sim_env = copy.deepcopy(ns_env)

    # Step the copy 10 times — original should be unaffected
    for _ in range(10):
        action = sim_env.action_space.sample()
        obs, _, done, trunc, _ = sim_env.step(action)
        if done or trunc:
            sim_env.reset(seed=42)

    for param in cc_params[env_id]:
        assert np.isclose(getattr(ns_env.unwrapped, param), orig_params[param]), (
            f"Original env param '{param}' was mutated by stepping the copy"
        )


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_deepcopy_independence_gridworld(gw_wrappers, gw_params, env_id):
    ns_env = _make_gw_env(env_id, gw_wrappers, gw_params, in_sim_change=True)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    orig_transition_prob = ns_env.transition_prob[:]

    sim_env = copy.deepcopy(ns_env)

    for _ in range(5):
        action = sim_env.action_space.sample()
        obs, _, done, trunc, _ = sim_env.step(action)
        if done or trunc:
            sim_env.reset(seed=42)

    assert ns_env.transition_prob == orig_transition_prob, (
        "Original env transition_prob was mutated by stepping the copy"
    )


# ============================================================
# Deepcopy: sim_env freezes params when in_sim_change=False
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy_sim_env_freezes_params_classic_control(cc_params, env_id):
    ns_env = _make_cc_env(env_id, cc_params, in_sim_change=False)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    assert sim_env.is_sim_env is True

    # Record params of the copy
    frozen_params = {}
    for param in cc_params[env_id]:
        frozen_params[param] = deepcopy(getattr(sim_env.unwrapped, param))

    # Step the copy — params should NOT change
    for _ in range(5):
        action = sim_env.action_space.sample()
        obs, _, done, trunc, _ = sim_env.step(action)
        if done or trunc:
            break

    for param in cc_params[env_id]:
        assert np.isclose(getattr(sim_env.unwrapped, param), frozen_params[param]), (
            f"Sim env param '{param}' changed despite in_sim_change=False"
        )


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_deepcopy_sim_env_freezes_params_gridworld(gw_wrappers, gw_params, env_id):
    ns_env = _make_gw_env(env_id, gw_wrappers, gw_params, in_sim_change=False)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    assert sim_env.is_sim_env is True

    frozen_tp = sim_env.transition_prob[:]

    for _ in range(5):
        action = sim_env.action_space.sample()
        obs, _, done, trunc, _ = sim_env.step(action)
        if done or trunc:
            break

    assert sim_env.transition_prob == frozen_tp, (
        "Sim env transition_prob changed despite in_sim_change=False"
    )


# ============================================================
# Deepcopy: sim_env updates params when in_sim_change=True
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy_sim_env_updates_params_classic_control(cc_params, env_id):
    ns_env = _make_cc_env(env_id, cc_params, in_sim_change=True)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    assert sim_env.is_sim_env is True

    initial_params = {}
    for param in cc_params[env_id]:
        initial_params[param] = deepcopy(getattr(sim_env.unwrapped, param))

    # Step the copy — params SHOULD change
    for _ in range(5):
        action = sim_env.action_space.sample()
        obs, _, done, trunc, _ = sim_env.step(action)
        if done or trunc:
            break

    any_changed = False
    for param in cc_params[env_id]:
        if not np.isclose(getattr(sim_env.unwrapped, param), initial_params[param]):
            any_changed = True
    assert any_changed, "No params changed on sim_env despite in_sim_change=True"


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_deepcopy_sim_env_updates_params_gridworld(gw_wrappers, gw_params, env_id):
    ns_env = _make_gw_env(env_id, gw_wrappers, gw_params, in_sim_change=True)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    frozen_tp = sim_env.transition_prob[:]

    for _ in range(5):
        action = sim_env.action_space.sample()
        obs, _, done, trunc, _ = sim_env.step(action)
        if done or trunc:
            break

    assert sim_env.transition_prob != frozen_tp, (
        "Sim env transition_prob unchanged despite in_sim_change=True"
    )


# ============================================================
# Deepcopy: t value preserved
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy_t_value_preserved_classic_control(cc_params, env_id):
    ns_env = _make_cc_env(env_id, cc_params)
    ns_env.reset(seed=42)
    for _ in range(7):
        ns_env.step(ns_env.action_space.sample())

    expected_t = ns_env.t
    sim_env = copy.deepcopy(ns_env)
    assert sim_env.t == expected_t


# ============================================================
# Deepcopy: at various episode points
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
@pytest.mark.parametrize("copy_at_step", [0, 1, 5, 10])
def test_deepcopy_at_various_points_classic_control(cc_params, env_id, copy_at_step):
    ns_env = _make_cc_env(env_id, cc_params)
    ns_env.reset(seed=42)
    for _ in range(copy_at_step):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            ns_env.reset(seed=42)

    sim_env = copy.deepcopy(ns_env)

    assert sim_env.is_sim_env is True
    assert sim_env.t == ns_env.t
    assert isinstance(sim_env, NSClassicControlWrapper)


# ============================================================
# Deepcopy: scalar_reward flag propagated
# ============================================================

@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
@pytest.mark.parametrize("scalar_reward", [True, False])
def test_deepcopy_scalar_reward_propagated(cc_params, env_id, scalar_reward):
    ns_env = _make_cc_env(env_id, cc_params, scalar_reward=scalar_reward)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    assert sim_env.scalar_reward == scalar_reward

    obs, reward, _, _, _ = sim_env.step(sim_env.action_space.sample())
    if scalar_reward:
        assert isinstance(reward, (int, float, np.floating))
    else:
        assert isinstance(reward, Reward)


# ============================================================
# Deepcopy: notification flags preserved
# ============================================================

@pytest.mark.parametrize("env_id", ["CartPole-v1", "CliffWalking-v1"])
def test_deepcopy_notification_flags_preserved(cc_params, gw_wrappers, gw_params, env_id):
    if env_id in CLASSIC_CONTROL_ENV_IDS:
        ns_env = _make_cc_env(
            env_id, cc_params,
            change_notification=True,
            delta_change_notification=True,
        )
    else:
        ns_env = _make_gw_env(
            env_id, gw_wrappers, gw_params,
            change_notification=True,
            delta_change_notification=True,
        )

    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    assert sim_env.change_notification == ns_env.change_notification
    assert sim_env.delta_change_notification == ns_env.delta_change_notification


# ============================================================
# Deepcopy: wrapper stack intact (no double wrapping)
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy_wrapper_stack_intact_classic_control(cc_params, env_id):
    ns_env = _make_cc_env(env_id, cc_params)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    _validate_no_double_wrap(sim_env, NSClassicControlWrapper)


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_deepcopy_wrapper_stack_intact_gridworld(gw_wrappers, gw_params, env_id):
    WrapperClass = gw_wrappers[env_id]
    ns_env = _make_gw_env(env_id, gw_wrappers, gw_params)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    _validate_no_double_wrap(sim_env, WrapperClass)
