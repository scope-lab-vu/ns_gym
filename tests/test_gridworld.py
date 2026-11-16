"""Behaviour specific to the FrozenLake and CliffWalking wrappers."""

import copy

import gymnasium as gym
import pytest

from ns_gym.base import UpdateFn
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import DistributionDecrementUpdate, DistributionIncrementUpdate
from ns_gym.wrappers import NSFrozenLakeWrapper

from _helpers import (
    CASE_BY_ID,
    GRIDWORLD_CASES,
    GRIDWORLD_ENV_IDS,
    env_cases,
    safe_actions,
    same,
    validate_no_double_wrap,
)


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_gridworld_wrapper_init(gw_wrappers, gw_params, env_id):
    """Tests the basic initialization of the gridworld wrappers."""
    env = gym.make(env_id)
    tunable_params = gw_params[env_id]
    WrapperClass = gw_wrappers[env_id]
    
    ns_env = WrapperClass(env, tunable_params, change_notification=True, delta_change_notification=True)
    
    assert isinstance(ns_env, WrapperClass)
    assert isinstance(ns_env.unwrapped, gym.Env)
    assert ns_env.change_notification
    assert ns_env.delta_change_notification
    assert len(ns_env.tunable_params) == len(tunable_params)
    assert all(isinstance(fn, UpdateFn) for fn in ns_env.tunable_params.values())


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_deepcopy_divergence(gw_wrappers, gw_params, env_id):
    """Tests that a deepcopied sim_env has a static transition model while the original evolves."""
    env = gym.make(env_id)
    tunable_params = gw_params[env_id]
    WrapperClass = gw_wrappers[env_id]
    
    ns_env = WrapperClass(env, tunable_params, change_notification=True, delta_change_notification=True)
    ns_env.reset(seed=42)
    
    action = ns_env.action_space.sample()

    obs_original, _, _, _, _ = ns_env.step(action)
    
    sim_env = copy.deepcopy(ns_env)

    assert isinstance(sim_env, WrapperClass)
    assert sim_env.is_sim_env
    assert sim_env.unwrapped.P.keys() == ns_env.unwrapped.P.keys()


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_get_planning_env(gw_wrappers, gw_params, env_id):
    """Tests that get_planning_env returns a deep copy with is_sim_env=True."""
    env = gym.make(env_id)
    tunable_params = gw_params[env_id]
    WrapperClass = gw_wrappers[env_id]
    
    ns_env = WrapperClass(env, tunable_params, change_notification=True, delta_change_notification=True)
    ns_env.reset(seed=42)

    done = True
    while done:
        obs,rew,done,truncated,info = ns_env.step(ns_env.action_space.sample())  # Take a step to change the environment
        if done:
            ns_env.reset()
    
    planning_env = ns_env.get_planning_env()
    validate_no_double_wrap(planning_env, WrapperClass)

    assert isinstance(planning_env, WrapperClass)
    assert planning_env.is_sim_env
    assert planning_env.unwrapped.P == ns_env.unwrapped.P
    assert planning_env.tunable_params.keys() == ns_env.tunable_params.keys()

    env = gym.make(env_id)
    tunable_params = gw_params[env_id]
    WrapperClass = gw_wrappers[env_id]
    ns_env = WrapperClass(env, tunable_params, change_notification=False, delta_change_notification=False)
    ns_env.reset(seed=42)

    done = True
    while done:
        obs,rew,done,truncated,info = ns_env.step(ns_env.action_space.sample())  # Take a step to change the environment
        if done:
            ns_env.reset()

    planning_env = ns_env.get_planning_env()
    validate_no_double_wrap(planning_env, WrapperClass)

    assert isinstance(planning_env, WrapperClass)
    assert planning_env.is_sim_env
    assert planning_env.unwrapped.P != ns_env.unwrapped.P


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_P_is_current_snapshot_not_future_schedule(
    gw_wrappers, env_id
):
    """``unwrapped.P`` must reflect the CURRENT non-stationary snapshot, not
    a precomputed schedule of future transitions.

    Specifically:
      * Repeated reads at the same time step return identical P.
      * P drifts only via ``step()``.
      * Probabilities at every (s, a) sum to 1 (no leakage of future-mass).
      * A deepcopy at time t freezes P; subsequent ``step()`` on the original
        does not change the snapshot's P.

    Note: this test deliberately uses ``DistributionDecrementUpdate`` (which
    clamps ``param[0]`` at 0) instead of the shared ``gw_params``
    fixture's ``DistributionIncrementUpdate(k=-0.1)``. The latter doesn't
    clamp negative parameters, so after ~10 steps it produces invalid
    distributions that the (now-correct) W1 rejects with
    ``ValueError: All weights must be non-negative.`` That's a real bug in
    ``DistributionIncrementUpdate`` -- but it's out of scope for *this* test,
    which is about snapshot semantics, not parameter validity.
    """
    from ns_gym.update_functions import DistributionDecrementUpdate
    env = gym.make(env_id)
    WrapperClass = gw_wrappers[env_id]
    # Decrement clamps at 0, so probs stay valid no matter how many steps.
    tunable_params = {
        "P": DistributionDecrementUpdate(ContinuousScheduler(), k=0.05),
    }
    ns_env = WrapperClass(env, tunable_params)
    ns_env.reset(seed=0)

    # Re-read stability
    P_a = copy.deepcopy(ns_env.unwrapped.P)
    P_b = copy.deepcopy(ns_env.unwrapped.P)
    assert P_a == P_b, "P read twice without stepping must be identical"

    # Drift via step
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
    P_t5 = copy.deepcopy(ns_env.unwrapped.P)
    assert P_a != P_t5, f"P must drift via step() (env={env_id})"

    # Mass conservation: every (s, a) entry must sum to 1
    bad = []
    for s in P_t5:
        for a in P_t5[s]:
            total = sum(t[0] for t in P_t5[s][a])
            if abs(total - 1.0) > 1e-9:
                bad.append((s, a, total))
    assert not bad, f"P[s][a] mass != 1 at: {bad[:3]}"

    # Deepcopy isolation: snapshot frozen against future drift
    snap = copy.deepcopy(ns_env)
    snap_P = copy.deepcopy(snap.unwrapped.P)
    for _ in range(10):
        ns_env.step(ns_env.action_space.sample())
    assert snap.unwrapped.P == snap_P, (
        f"deepcopy leaked future drift (env={env_id})"
    )
    assert ns_env.unwrapped.P != snap_P, (
        f"original env did not drift after snapshot (env={env_id})"
    )


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_invalid_tunable_param_gridworld(env_id, gw_wrappers):
    """Tests that the wrapper raises an error for an invalid parameter name."""
    env = gym.make(env_id)
    update_fn = DistributionIncrementUpdate(ContinuousScheduler(), k=0.1)
    tunable_params = {"invalid_param": update_fn} # Invalid parameter

    with pytest.raises(AssertionError):
        ns_env = gw_wrappers[env_id](env, tunable_params)




# ============================================================
# Reported probabilities are the ones the env samples from
# ============================================================

@pytest.mark.parametrize("case", env_cases(GRIDWORLD_CASES))
def test_reported_probabilities_match_executed_table(case):
    """``transition_prob`` (also in ``info``) equals the table of the base env after every step."""
    env = case.make()
    env.reset(seed=0)
    assert same(case.reported(env), case.executed(env))

    for action in safe_actions(case, env, 5):
        _, _, _, _, info = env.step(action)
        assert same(info["transition_prob"], case.executed(env))
        assert same(case.reported(env), case.executed(env))


@pytest.mark.parametrize("case", env_cases(GRIDWORLD_CASES))
def test_transition_table_rows_sum_to_one(case):
    env = case.make()
    env.reset(seed=0)
    for action in safe_actions(case, env, 5):
        env.step(action)

    P = env.unwrapped.P
    bad = [(s, a) for s in P for a in P[s] if abs(sum(t[0] for t in P[s][a]) - 1.0) > 1e-9]
    assert not bad, f"P[s][a] does not sum to 1 at {bad[:3]}"


# ============================================================
# FrozenLake options
# ============================================================

def _frozenlake(update_fn=None, make_kwargs=None, **wrapper_kwargs):
    case = CASE_BY_ID["FrozenLake-v1"]
    env = case.make(update_fn or case.no_update_fn(), make_kwargs=make_kwargs, **wrapper_kwargs)
    env.reset(seed=0)
    return env


def test_frozenlake_default_rewards():
    env = _frozenlake()
    rewards = [env.step(a)[1] for a in [1, 1, 2, 1, 2, 2]]  # shortest safe path to the goal
    assert rewards == [0, 0, 0, 0, 0, 1]


def test_frozenlake_modified_rewards():
    modified = {"H": -1, "G": 10, "F": -0.1, "S": -0.1}
    env = _frozenlake(modified_rewards=modified)
    rewards, terminated = [], False
    for action in [1, 1, 2, 1, 2, 2]:
        _, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)
    assert rewards == [-0.1, -0.1, -0.1, -0.1, -0.1, 10]
    assert terminated

    env = _frozenlake(modified_rewards=modified)
    env.step(2)  # 0 -> 1
    _, reward, terminated, _, _ = env.step(1)  # 1 -> 5, a hole
    assert (reward, terminated) == (-1, True)


def test_frozenlake_8x8_map():
    env = _frozenlake(make_kwargs={"map_name": "8x8"})
    assert (env.nrow, env.ncol, env.nS) == (8, 8, 64)
    assert len(env.unwrapped.P) == 64
    states = [env.step(2)[0]["state"] for _ in range(7)]  # right along the top row
    assert states == [1, 2, 3, 4, 5, 6, 7]


def test_frozenlake_custom_map():
    desc = ["SFFFF", "FHFHF", "FFFHF", "HFFFG"]
    env = _frozenlake(make_kwargs={"desc": desc})
    assert (env.nrow, env.ncol, env.nS) == (4, 5, 20)
    states = [env.step(2)[0]["state"] for _ in range(4)]
    assert states == [1, 2, 3, 4]


@pytest.mark.parametrize("bad_dist", [[0.5, 0.2, 0.2], [0.5, 0.5], [0.25, 0.25, 0.25, 0.25]])
def test_frozenlake_rejects_invalid_initial_distribution(bad_dist):
    fn = DistributionDecrementUpdate(ContinuousScheduler(), k=0.1)
    with pytest.raises(AssertionError):
        NSFrozenLakeWrapper(gym.make("FrozenLake-v1"), {"P": fn}, initial_prob_dist=bad_dist)


def test_frozenlake_slip_goes_perpendicular():
    """With ``[0, 1, 0]`` every move is replaced by the action one step clockwise."""
    env = _frozenlake(initial_prob_dist=[0.0, 1.0, 0.0])
    obs, _, _, _, _ = env.step(0)  # LEFT intended -> (a + 1) % 4 = DOWN
    assert obs["state"] == 4


# ============================================================
# CliffWalking options
# ============================================================

def _cliffwalking(**wrapper_kwargs):
    case = CASE_BY_ID["CliffWalking-v1"]
    env = case.make(case.no_update_fn(), **wrapper_kwargs)
    env.reset(seed=0)
    return env


def test_cliffwalking_default_cliff_returns_to_start():
    env = _cliffwalking()
    obs, reward, terminated, _, _ = env.step(1)  # right, off the cliff
    assert (obs["state"], reward, terminated) == (env.start_state_index, -100, False)


def test_cliffwalking_terminal_cliff():
    env = _cliffwalking(terminal_cliff=True)
    _, reward, terminated, _, _ = env.step(1)
    assert (reward, terminated) == (-100, True)


def test_cliffwalking_modified_rewards():
    env = _cliffwalking(modified_rewards={"H": -7, "G": 5, "F": -2, "S": -2})
    _, reward, _, _, _ = env.step(0)  # up, an ordinary cell
    assert reward == -2
    env.step(2)  # back down to the start
    _, reward, _, _, _ = env.step(1)  # right, off the cliff
    assert reward == -7


def test_cliffwalking_reaches_goal():
    env = _cliffwalking()
    path = [0] + [1] * 11 + [2]  # up, along the row above the cliff, down onto the goal
    terminated = False
    for action in path:
        obs, reward, terminated, _, _ = env.step(action)
    assert terminated and obs["state"] == 47
