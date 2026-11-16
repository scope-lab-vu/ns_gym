"""Copies and planning environments for the classic-control and gridworld wrappers."""

import copy
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

from ns_gym.base import TUNABLE_PARAMS, Reward
from ns_gym.schedulers import ContinuousScheduler, DiscreteScheduler
from ns_gym.update_functions import RandomCategorical, RandomWalk
from ns_gym.wrappers import NSClassicControlWrapper

from _helpers import (
    CASE_BY_ID,
    CC_SEEDING_PARAMS,
    CLASSIC_CONTROL_ENV_IDS,
    DISTRIBUTION_CASES,
    GRIDWORLD_ENV_IDS,
    env_cases,
    known_bug,
    safe_actions,
    same,
    state_of,
    validate_no_double_wrap,
)


def _make_cc_env(env_id, cc_params, **wrapper_kwargs):
    env = gym.make(env_id)
    return NSClassicControlWrapper(env, cc_params[env_id], **wrapper_kwargs)


def _make_gw_env(env_id, gw_wrappers, gw_params, **wrapper_kwargs):
    env = gym.make(env_id)
    WrapperClass = gw_wrappers[env_id]
    return WrapperClass(env, gw_params[env_id], **wrapper_kwargs)


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
    validate_no_double_wrap(sim_env, NSClassicControlWrapper)


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_deepcopy_wrapper_stack_intact_gridworld(gw_wrappers, gw_params, env_id):
    WrapperClass = gw_wrappers[env_id]
    ns_env = _make_gw_env(env_id, gw_wrappers, gw_params)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    validate_no_double_wrap(sim_env, WrapperClass)


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_planning_env_rng_diverges_classic_control(env_id):
    """Planning env should NOT predict future stochastic parameter changes."""
    param = CC_SEEDING_PARAMS[env_id]
    env = gym.make(env_id)
    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.1, seed=42)
    ns_env = NSClassicControlWrapper(
        env, {param: fn},
        change_notification=True,
        delta_change_notification=True,
    )
    ns_env.reset(seed=0)

    # Step a few times to advance state
    for _ in range(3):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            ns_env.reset(seed=0)

    planning_env = ns_env.get_planning_env()

    # --- Explicit RNG state comparison ---
    real_fn = ns_env.tunable_params[param]
    plan_fn = planning_env.tunable_params[param]
    if hasattr(real_fn, 'rng') and hasattr(plan_fn, 'rng'):
        real_sample = real_fn.rng.normal()
        plan_sample = plan_fn.rng.normal()
        assert real_sample != plan_sample, (
            f"[{env_id}] Planning env RNG should be independent from real env"
        )

    # --- Trajectory divergence with same action sequence ---
    # Re-create to get fresh RNG (previous comparison consumed a sample)
    env = gym.make(env_id)
    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.1, seed=42)
    ns_env = NSClassicControlWrapper(
        env, {param: fn},
        change_notification=True,
        delta_change_notification=True,
    )
    ns_env.reset(seed=0)
    for _ in range(3):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            ns_env.reset(seed=0)

    planning_env = ns_env.get_planning_env()

    # Use a fixed action for both envs
    action = 0 if hasattr(ns_env.action_space, 'n') else np.zeros(ns_env.action_space.shape)
    real_trajectory = []
    planning_trajectory = []
    for _ in range(10):
        obs_real, _, done, trunc, _ = ns_env.step(action)
        if done or trunc:
            break
        real_trajectory.append(getattr(env.unwrapped, param))

        obs_plan, _, _, _, _ = planning_env.step(action)
        planning_trajectory.append(getattr(planning_env.unwrapped, param))

    assert len(real_trajectory) > 0, f"[{env_id}] No steps completed"
    assert real_trajectory != planning_trajectory, (
        f"[{env_id}] Planning env should have independent RNG — trajectories must diverge"
    )


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_planning_env_rng_diverges_gridworld(gw_wrappers, env_id):
    """Gridworld planning env should NOT predict future stochastic parameter changes."""

    env = gym.make(env_id)
    WrapperClass = gw_wrappers[env_id]
    fn = RandomCategorical(ContinuousScheduler(), seed=42)
    ns_env = WrapperClass(
        env, {"P": fn},
        change_notification=True,
        delta_change_notification=True,
    )
    ns_env.reset(seed=0)

    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    planning_env = ns_env.get_planning_env()

    # --- Explicit RNG state comparison ---
    real_fn = ns_env.tunable_params["P"]
    plan_fn = planning_env.tunable_params["P"]
    if hasattr(real_fn, 'rng') and hasattr(plan_fn, 'rng'):
        real_sample = real_fn.rng.random()
        plan_sample = plan_fn.rng.random()
        assert real_sample != plan_sample, (
            f"[{env_id}] Planning env RNG should be independent from real env"
        )

    # --- Trajectory divergence with same action sequence ---
    # Re-create to get fresh RNG (previous comparison consumed a sample)
    env = gym.make(env_id)
    fn = RandomCategorical(ContinuousScheduler(), seed=42)
    ns_env = WrapperClass(
        env, {"P": fn},
        change_notification=True,
        delta_change_notification=True,
    )
    ns_env.reset(seed=0)

    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    planning_env = ns_env.get_planning_env()

    # Use the same action sequence for both envs
    actions = [ns_env.action_space.sample() for _ in range(5)]
    for action in actions:
        obs_real, _, done_real, trunc_real, _ = ns_env.step(action)
        obs_plan, _, done_plan, trunc_plan, _ = planning_env.step(action)
        if done_real or trunc_real or done_plan or trunc_plan:
            break

    assert ns_env.transition_prob != planning_env.transition_prob, (
        f"[{env_id}] Planning env should have independent RNG — transition probs must diverge"
    )


# ============================================================
# Every supported env: what a planning env must look like
# ============================================================

@pytest.mark.parametrize("case", env_cases())
def test_planning_env_structure(case):
    """A planning env is a separate, flagged copy with the same spaces and time."""
    env = case.make(change_notification=True, delta_change_notification=True)
    env.reset(seed=0)
    env.step(safe_actions(case, env, 1)[0])

    planning_env = env.get_planning_env()

    assert planning_env is not env
    assert planning_env.is_sim_env is True
    assert env.is_sim_env is False
    validate_no_double_wrap(planning_env, case.wrapper)
    assert planning_env.t == env.t
    assert planning_env.action_space == env.action_space
    assert planning_env.unwrapped.observation_space == env.unwrapped.observation_space
    assert planning_env.observation_space == env.observation_space


@pytest.mark.parametrize("case", env_cases())
def test_planning_env_with_delta_notification_has_current_parameter(case):
    """An agent told the size of each change plans with the current dynamics."""
    env = case.make(change_notification=True, delta_change_notification=True)
    env.reset(seed=0)
    for action in safe_actions(case, env, 2):
        env.step(action)
    assert not same(case.executed(env), case.initial())

    assert same(case.executed(env.get_planning_env()), case.executed(env))


@pytest.mark.parametrize("case", env_cases())
def test_planning_env_without_notification_has_initial_parameter(case):
    """An agent that is told nothing plans with the dynamics it started with."""
    env = case.make()
    env.reset(seed=0)
    for action in safe_actions(case, env, 2):
        env.step(action)
    assert not same(case.executed(env), case.initial())

    assert same(case.executed(env.get_planning_env()), case.initial())


@pytest.mark.parametrize("case", env_cases())
def test_planning_env_parameter_is_frozen(case):
    """Stepping a planning env does not change its parameter (``in_sim_change=False``)."""
    env = case.make(change_notification=True, delta_change_notification=True)
    env.reset(seed=0)
    env.step(safe_actions(case, env, 1)[0])
    planning_env = env.get_planning_env()
    frozen = case.executed(planning_env)

    for action in safe_actions(case, planning_env, 3):
        obs, _, _, _, _ = planning_env.step(action)
        assert obs["env_change"] == {case.param: 0}
    assert same(case.executed(planning_env), frozen)


@pytest.mark.parametrize("case", env_cases())
def test_planning_env_follows_real_env_when_stationary(case):
    """With no update and deterministic dynamics, the copy and the real env
    visit the same states and collect the same rewards.

    Rewards are part of the check because Ant and Humanoid compute a velocity
    reward from body positions in the simulation data, so the copy must carry
    the same derived quantities as the original, not only the same state.
    """
    env = case.make(case.no_update_fn(), change_notification=True, delta_change_notification=True)
    env.reset(seed=0)
    actions = safe_actions(case, env, 6)
    for action in actions[:3]:
        env.step(action)

    planning_env = env.get_planning_env()
    for action in actions[3:]:
        obs_r, rew_r, term_r, _, _ = env.step(action)
        obs_p, rew_p, term_p, _, _ = planning_env.step(action)
        assert np.allclose(state_of(obs_r), state_of(obs_p)), "states diverged"
        assert np.isclose(rew_r, rew_p), "rewards diverged"
        assert term_r == term_p, "termination diverged"
        if term_r:
            break


@pytest.mark.parametrize("case", env_cases())
def test_stepping_planning_env_leaves_real_env_untouched(case):
    env = case.make(change_notification=True, delta_change_notification=True)
    env.reset(seed=0)
    env.step(safe_actions(case, env, 1)[0])
    executed, t = case.executed(env), env.t

    planning_env = env.get_planning_env()
    for action in safe_actions(case, planning_env, 3):
        planning_env.step(action)

    assert env.t == t
    assert same(case.executed(env), executed)


# ============================================================
# Gridworld planning envs carry the real env's map and table
# ============================================================

@pytest.mark.parametrize("case", env_cases(DISTRIBUTION_CASES))
def test_planning_env_taken_after_reset_has_initial_dynamics(case):
    """A planner asking for its model at the start of episode 2 gets episode 2's dynamics."""
    env = case.make(
        scheduler=DiscreteScheduler({1}),
        change_notification=True,
        delta_change_notification=True,
    )
    env.reset(seed=0)
    for action in safe_actions(case, env, 3):
        env.step(action)
    assert not same(case.executed(env), case.initial())

    env.reset(seed=1)
    assert same(case.executed(env.get_planning_env()), case.initial())


FROZENLAKE_MAPS = {
    "4x4": {},
    "8x8": {"map_name": "8x8"},
    "custom4x5": {"desc": ["SFFFF", "FHFHF", "FFFHF", "HFFFG"]},
}


@pytest.mark.parametrize("in_sim_change", [False, True], ids=["normal", "oracle"])
@pytest.mark.parametrize("map_id", ["4x4", "8x8", "custom4x5"])
def test_frozenlake_planning_env_has_the_real_map(map_id, in_sim_change):
    """The planning env describes, and moves on, the same grid as the real env."""
    case = CASE_BY_ID["FrozenLake-v1"]

    def make():
        env = case.make(
            scheduler=DiscreteScheduler({50}),
            make_kwargs=FROZENLAKE_MAPS[map_id],
            change_notification=True,
            delta_change_notification=True,
            in_sim_change=in_sim_change,
        )
        env.reset(seed=0)
        return env

    real, planning = make(), make().get_planning_env()

    assert np.array_equal(planning.unwrapped.desc, real.unwrapped.desc)
    assert planning.unwrapped.observation_space == real.unwrapped.observation_space
    assert (planning.nrow, planning.ncol, planning.nS) == (real.nrow, real.ncol, real.nS)

    for action in [2] * 7 + [1] * 3:  # right along the top row, then down
        obs_r, rew_r, term_r, _, _ = real.step(action)
        obs_p, rew_p, term_p, _, _ = planning.step(action)
        assert (obs_r["state"], rew_r, term_r) == (obs_p["state"], rew_p, term_p)
        if term_r:
            break


def test_frozenlake_oracle_planning_env_keeps_current_table():
    """Oracle mode (``in_sim_change=True``): the copy keeps stepping with the
    table that was current when it was taken, not the initial one."""
    case = CASE_BY_ID["FrozenLake-v1"]
    env = case.make(
        scheduler=DiscreteScheduler({1}),
        change_notification=True,
        delta_change_notification=True,
        in_sim_change=True,
    )
    env.reset(seed=0)
    for action in safe_actions(case, env, 3):
        env.step(action)

    planning_env = env.get_planning_env()
    assert same(case.executed(planning_env), case.executed(env))
    planning_env.step(0)
    assert same(case.executed(planning_env), case.executed(env))
    assert same(case.reported(planning_env), case.executed(planning_env))


@known_bug("C3")
def test_cliffwalking_planning_env_keeps_reward_options():
    rewards = {"H": -7, "G": 5, "F": -2, "S": -2}
    env = CASE_BY_ID["CliffWalking-v1"].make(modified_rewards=rewards, terminal_cliff=True)
    env.reset(seed=0)
    planning_env = env.get_planning_env()
    assert planning_env.modified_rewards == rewards
    assert planning_env.terminal_cliff is True


def test_cliffwalking_planning_env_keeps_reward_dynamics():
    """Whatever the copy's attributes say, it must pay and terminate like the real env."""
    rewards = {"H": -7, "G": 5, "F": -2, "S": -2}
    case = CASE_BY_ID["CliffWalking-v1"]
    env = case.make(case.no_update_fn(), modified_rewards=rewards, terminal_cliff=True)
    env.reset(seed=0)
    planning_env = env.get_planning_env()

    _, reward, terminated, _, _ = planning_env.step(1)  # right, off the cliff
    assert (reward, terminated) == (-7, True)


# ============================================================
# Planning envs keep what the real env was built with
# ============================================================

def _cartpole_is_consistent(unwrapped):
    return bool(
        np.isclose(unwrapped.total_mass, unwrapped.masspole + unwrapped.masscart)
        and np.isclose(unwrapped.polemass_length, unwrapped.masspole * unwrapped.length)
    )


@pytest.mark.parametrize(
    "notify", [pytest.param(False, marks=known_bug("C1")), True], ids=["no-notification", "notification"]
)
def test_cartpole_planning_env_derived_quantities(notify):
    """``total_mass`` and ``polemass_length`` must follow whatever ``masspole`` the copy has."""
    case = CASE_BY_ID["CartPole-v1"]
    env = case.make(change_notification=notify, delta_change_notification=notify)
    env.reset(seed=0)
    for action in safe_actions(case, env, 4):
        env.step(action)

    assert _cartpole_is_consistent(env.get_planning_env().unwrapped)


CONSTRUCTOR_ARGUMENTS = [
    ("Pendulum-v1", {"g": 3.0}, lambda u: u.g),
    ("CartPole-v1", {"sutton_barto_reward": True}, lambda u: u._sutton_barto_reward),
    ("MountainCar-v0", {"goal_velocity": 0.02}, lambda u: u.goal_velocity),
    ("Hopper-v5", {"terminate_when_unhealthy": False}, lambda u: u._terminate_when_unhealthy),
    ("Hopper-v5", {"forward_reward_weight": 5.0}, lambda u: u._forward_reward_weight),
]


@known_bug("C2")
@pytest.mark.parametrize(
    "env_id,make_kwargs,read",
    CONSTRUCTOR_ARGUMENTS,
    ids=[f"{env_id}-{next(iter(kwargs))}" for env_id, kwargs, _ in CONSTRUCTOR_ARGUMENTS],
)
def test_planning_env_keeps_constructor_arguments(env_id, make_kwargs, read):
    """Options passed to ``gym.make`` for the real env apply to its planning env too."""
    env = CASE_BY_ID[env_id].make(make_kwargs=make_kwargs)
    env.reset(seed=0)
    assert read(env.unwrapped) == next(iter(make_kwargs.values())), "the real env ignored the argument"

    assert read(env.get_planning_env().unwrapped) == read(env.unwrapped)
