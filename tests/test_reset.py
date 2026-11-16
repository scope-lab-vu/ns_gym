"""What ``reset()`` restores, and what ``persistent_params`` keeps."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

from ns_gym.base import TUNABLE_PARAMS
from ns_gym.schedulers import ContinuousScheduler, DiscreteScheduler
from ns_gym.update_functions import IncrementUpdate, RandomWalk
from ns_gym.wrappers import MujocoWrapper, NSClassicControlWrapper
from ns_gym.wrappers.mujoco_env import param_look_up

from _helpers import (
    CLASSIC_CONTROL_ENV_IDS,
    GRIDWORLD_ENV_IDS,
    MUJOCO_TIME_TEST_IDS,
    OBS_KEYS,
    env_cases,
    safe_actions,
    same,
)


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


def test_persistent_params_preserves_values_classic_control():
    """With persistent_params=True, params should NOT reset to initial values."""
    env = gym.make("CartPole-v1")
    fn = IncrementUpdate(ContinuousScheduler(), k=0.5)
    ns_env = NSClassicControlWrapper(env, {"masspole": fn}, persistent_params=True)
    ns_env.reset(seed=42)

    initial_masspole = env.unwrapped.masspole

    # Step to mutate masspole
    for _ in range(5):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            break

    mutated_masspole = env.unwrapped.masspole
    assert mutated_masspole != initial_masspole, "masspole should have changed after steps"

    # Reset — params should persist
    ns_env.reset(seed=42)
    assert ns_env.t == 0, "Time should still reset to 0"
    assert env.unwrapped.masspole == mutated_masspole, (
        f"masspole should persist after reset: expected {mutated_masspole}, "
        f"got {env.unwrapped.masspole}"
    )


def test_persistent_params_rng_continuity_classic_control():
    """With persistent_params=True and no seed, RNG continues across resets."""
    env = gym.make("CartPole-v1")
    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.01, seed=42)
    ns_env = NSClassicControlWrapper(env, {"masspole": fn}, persistent_params=True)
    ns_env.reset(seed=0)

    trajectory_a = []
    for _ in range(5):
        ns_env.step(0)
        trajectory_a.append(env.unwrapped.masspole)

    # Reset WITHOUT seed — RNG should continue (persistent_params=True)
    ns_env.reset()
    trajectory_b = []
    for _ in range(5):
        ns_env.step(0)
        trajectory_b.append(env.unwrapped.masspole)

    assert trajectory_a != trajectory_b, (
        "With persistent_params=True and no seed, RNG should continue, not restart"
    )


def test_persistent_params_default_false_unchanged(cc_params):
    """Explicitly passing persistent_params=False should behave like default (params restored)."""
    env = gym.make("CartPole-v1")
    ns_env = NSClassicControlWrapper(
        env, cc_params["CartPole-v1"], persistent_params=False
    )
    ns_env.reset(seed=42)

    initial_masspole = env.unwrapped.masspole

    for _ in range(5):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            break

    ns_env.reset(seed=42)
    assert np.isclose(env.unwrapped.masspole, initial_masspole), (
        "With persistent_params=False, masspole should be restored after reset"
    )


@pytest.mark.parametrize("env_id", MUJOCO_TIME_TEST_IDS)
def test_persistent_params_preserves_values_mujoco(env_id):
    """With persistent_params=True, MuJoCo params should NOT reset to initial values."""

    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    # Use 'pole_mass' for InvertedPendulum, first non-gravity param otherwise
    params = list(TUNABLE_PARAMS[env_name].keys())
    param_name = next((p for p in params if p != "gravity"), params[0])

    fn = IncrementUpdate(ContinuousScheduler(), k=0.5)
    ns_env = MujocoWrapper(env, {param_name: fn}, persistent_params=True)
    ns_env.reset(seed=42)

    getter, _ = param_look_up(env_name, param_name)[0]
    initial_value = getter(env.unwrapped)

    for _ in range(3):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            break

    mutated_value = getter(env.unwrapped)
    assert not np.allclose(mutated_value, initial_value), (
        f"Parameter '{param_name}' should have changed after steps"
    )

    ns_env.reset(seed=42)
    assert ns_env.t == 0, "Time should still reset to 0"

    value_after_reset = getter(env.unwrapped)
    assert np.allclose(value_after_reset, mutated_value), (
        f"With persistent_params=True, '{param_name}' should persist after reset: "
        f"expected {mutated_value}, got {value_after_reset}"
    )


# ============================================================
# Every supported env: reset restores the dynamics the env executes
# ============================================================

@pytest.mark.parametrize("case", env_cases())
def test_reset_restores_executed_parameter(case):
    """After an episode with updates, ``reset()`` puts the base env's own value back."""
    env = case.make()
    env.reset(seed=42)
    assert same(case.executed(env), case.initial())

    for action in safe_actions(case, env, 3):
        env.step(action)
    assert not same(case.executed(env), case.initial()), "the update never changed the parameter"

    env.reset(seed=42)
    assert env.t == 0
    assert same(case.executed(env), case.initial())


@pytest.mark.parametrize("case", env_cases())
def test_every_episode_starts_with_initial_dynamics(case):
    """A change scheduled mid-episode must not leak into the start of the next one.

    The scheduler fires only at step 2. Checking the executed value *after the
    first step* is what distinguishes this from the test above: the gridworld
    wrappers keep their own copy of the transition table and re-install it on
    every ``step()``, so restoring only the base env on reset is not enough
    (issue #44).
    """
    env = case.make(scheduler=DiscreteScheduler({2}))

    for episode in range(3):
        env.reset(seed=episode)
        actions = safe_actions(case, env, 3)
        assert same(case.executed(env), case.initial())

        env.step(actions[0])  # t = 0: nothing scheduled
        assert same(case.executed(env), case.initial()), (
            f"episode {episode}: dynamics at step 0 are not the initial ones"
        )
        assert same(case.reported(env), case.executed(env)), (
            f"episode {episode}: reported value differs from the one the env executes"
        )

        env.step(actions[1])  # t = 1
        env.step(actions[2])  # t = 2: the change fires
        assert not same(case.executed(env), case.initial()), (
            f"episode {episode}: the scheduled change did not happen"
        )


@pytest.mark.parametrize("case", env_cases())
def test_persistent_params_keep_executed_parameter(case):
    """With ``persistent_params=True`` a reset keeps the changed dynamics and restarts time."""
    env = case.make(scheduler=DiscreteScheduler({1}), persistent_params=True)
    env.reset(seed=0)
    for action in safe_actions(case, env, 3):
        env.step(action)
    changed = case.executed(env)
    assert not same(changed, case.initial())

    env.reset(seed=1)
    assert env.t == 0
    assert same(case.executed(env), changed)

    env.step(safe_actions(case, env, 1)[0])  # t = 0 again: nothing scheduled
    assert same(case.executed(env), changed)
