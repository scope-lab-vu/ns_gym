"""The interface every ns_gym wrapper shares, checked on every supported env."""

import gymnasium as gym
import numpy as np
import pytest

from ns_gym.base import Reward
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate
from ns_gym.wrappers import NSClassicControlWrapper

from _helpers import (
    CASE_BY_ID,
    OBS_KEYS,
    env_cases,
    safe_actions,
    same,
    validate_no_double_wrap,
)


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
# Every supported env: observation, time counter, reward, info
# ============================================================

@pytest.mark.parametrize("case", env_cases())
def test_wrapper_is_outermost_and_applied_once(case):
    validate_no_double_wrap(case.make(), case.wrapper)


@pytest.mark.parametrize("case", env_cases())
def test_reset_observation(case):
    """``reset()`` returns the ns_gym observation dict with no change reported."""
    env = case.make(change_notification=True, delta_change_notification=True)
    obs, info = env.reset(seed=42)

    assert isinstance(obs, dict)
    assert set(obs) == set(OBS_KEYS)
    assert obs["relative_time"] == 0
    assert obs["env_change"] == {case.param: 0}
    assert obs["delta_change"] == {case.param: 0}
    assert env.observation_space.contains(obs), "reset observation is outside observation_space"
    assert info["Ground Truth Env Change"] == {case.param: 0}
    assert info["Ground Truth Delta Change"] == {case.param: 0}


@pytest.mark.parametrize("case", env_cases())
def test_step_observation(case):
    """``step()`` reports the change when both notification levels are on."""
    env = case.make(change_notification=True, delta_change_notification=True)
    env.reset(seed=42)
    obs, _, _, _, _ = env.step(safe_actions(case, env, 1)[0])

    assert isinstance(obs, dict)
    assert set(obs) == set(OBS_KEYS)
    assert obs["relative_time"] == 1
    assert obs["env_change"] == {case.param: 1}
    assert obs["delta_change"][case.param] != 0
    assert env.observation_space.contains(obs), "step observation is outside observation_space"
    if isinstance(env.unwrapped.observation_space, gym.spaces.Discrete):
        assert isinstance(obs["state"], int)


@pytest.mark.parametrize("case", env_cases())
def test_time_counter_increments_by_one(case):
    """``t`` and ``relative_time`` advance by exactly one per step."""
    env = case.make()
    env.reset(seed=42)
    assert env.t == 0

    for step_num, action in enumerate(safe_actions(case, env, 5), start=1):
        obs, _, _, _, _ = env.step(action)
        assert env.t == step_num, f"Expected t={step_num}, got t={env.t}"
        assert obs["relative_time"] == step_num


@pytest.mark.parametrize("case", env_cases())
@pytest.mark.parametrize("scalar_reward", [True, False], ids=["scalar", "dataclass"])
def test_reward_type(case, scalar_reward):
    """``scalar_reward`` selects a plain number or the ``Reward`` dataclass."""
    env = case.make(scalar_reward=scalar_reward)
    env.reset(seed=42)
    _, reward, _, _, _ = env.step(safe_actions(case, env, 1)[0])

    if scalar_reward:
        assert isinstance(reward, (int, float, np.integer, np.floating)), (
            f"Expected scalar reward, got {type(reward)}"
        )
    else:
        assert isinstance(reward, Reward), f"Expected Reward dataclass, got {type(reward)}"
        assert isinstance(reward.reward, (int, float, np.integer, np.floating))
        assert reward.env_change == {case.param: 0}
        assert reward.delta_change == {case.param: 0.0}
        assert reward.relative_time == 1


@pytest.mark.parametrize("case", env_cases())
def test_ground_truth_in_info_without_notifications(case):
    """With notifications off the observation hides the change; ``info`` keeps it."""
    env = case.make(change_notification=False, delta_change_notification=False)
    env.reset(seed=42)
    obs, _, _, _, info = env.step(safe_actions(case, env, 1)[0])

    assert obs["env_change"] == {case.param: 0}
    assert obs["delta_change"] == {case.param: 0.0}
    assert info["Ground Truth Env Change"] == {case.param: 1}
    assert info["Ground Truth Delta Change"][case.param] != 0


@pytest.mark.parametrize("case", env_cases())
def test_unknown_parameter_is_rejected(case):
    with pytest.raises(AssertionError):
        case.wrapper(gym.make(case.env_id), {"not_a_parameter": case.update_fn()})


@pytest.mark.parametrize("case", env_cases())
def test_delta_notification_requires_change_notification(case):
    with pytest.raises(AssertionError):
        case.make(change_notification=False, delta_change_notification=True)


@pytest.mark.parametrize("case", env_cases())
def test_planning_env_requires_reset(case):
    env = case.make()
    with pytest.raises(AssertionError):
        env.get_planning_env()


# ============================================================
# freeze()
# ============================================================

def test_freeze_hides_notifications():
    case = CASE_BY_ID["Pendulum-v1"]
    env = case.make(change_notification=True, delta_change_notification=True)
    env.reset(seed=0)
    env.freeze()
    obs, _, _, _, _ = env.step(safe_actions(case, env, 1)[0])
    assert obs["env_change"] == {case.param: 0}
    assert obs["delta_change"] == {case.param: 0.0}

    env.unfreeze()
    obs, _, _, _, _ = env.step(safe_actions(case, env, 1)[0])
    assert obs["env_change"] == {case.param: 1}


@pytest.mark.parametrize("case", env_cases())
def test_freeze_stops_parameter_updates(case):
    """While frozen the dynamics do not change; they resume after ``unfreeze()``."""
    env = case.make()
    env.reset(seed=0)
    actions = safe_actions(case, env, 5)
    env.step(actions[0])
    before = case.executed(env)

    env.freeze()
    for action in actions[1:4]:
        env.step(action)
    assert same(case.executed(env), before)
    assert env.t == 4, "time keeps advancing while frozen"

    env.unfreeze()
    env.step(actions[4])
    assert not same(case.executed(env), before)
