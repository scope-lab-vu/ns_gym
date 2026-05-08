"""Tests for the Bridge env and its NSBridgeWrapper.

Bridge differs from FrozenLake/CliffWalking in two important ways:
  * ``unwrapped.P`` is a list (the slip 3-vector), not a transition-table dict.
  * It supports a "split mode" where the left and right halves of the map
    carry independent slip distributions (``P_left``, ``P_right``).

Tests in this file cover both regimes and include regressions for the
historical "property 'P' has no setter" bug.
"""

import copy
import gymnasium as gym
import numpy as np
import pytest

import ns_gym
from ns_gym.base import TUNABLE_PARAMS, UpdateFn
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import (
    DistributionNoUpdate,
    UniformDrift,
    DistributionIncrementUpdate,
)
from ns_gym.wrappers import NSBridgeWrapper


# ---------------------------------------------------------------------------
# Bare-env tests
# ---------------------------------------------------------------------------

@pytest.fixture
def bridge_env():
    return gym.make("ns_gym/Bridge-v0")


def test_bridge_registered_in_gym(bridge_env):
    """ns_gym/Bridge-v0 must be a valid registered gym env."""
    assert bridge_env is not None
    assert hasattr(bridge_env.unwrapped, "step")
    assert hasattr(bridge_env.unwrapped, "reset")


def test_bridge_default_attributes(bridge_env):
    """Default slip distributions should be lists (not properties), and
    the legacy ``left_side_prob`` / ``right_side_prob`` aliases must
    exist for back-compat with older code."""
    inner = bridge_env.unwrapped
    assert isinstance(inner.P, list)
    assert isinstance(inner.P_left, list)
    assert isinstance(inner.P_right, list)
    assert inner.left_side_prob is inner.P_left
    assert inner.right_side_prob is inner.P_right
    assert inner.split_probs is False


def test_bridge_P_setter_does_not_raise(bridge_env):
    """Regression test for the historical bug:
        AttributeError: property 'P' of 'Bridge' object has no setter
    """
    inner = bridge_env.unwrapped
    new = [0.7, 0.2, 0.1]
    inner.P = new                     # plain assignment
    assert inner.P == new
    setattr(inner, "P", [0.5, 0.25, 0.25])   # via setattr (used by wrapper)
    assert inner.P == [0.5, 0.25, 0.25]


def test_bridge_P_left_and_P_right_setters(bridge_env):
    inner = bridge_env.unwrapped
    inner.P_left = [0.9, 0.05, 0.05]
    inner.P_right = [0.4, 0.3, 0.3]
    assert inner.P_left == [0.9, 0.05, 0.05]
    assert inner.P_right == [0.4, 0.3, 0.3]


def test_bridge_transition_matrix_is_dict(bridge_env):
    """``transition_matrix`` must return P[s][a] = list of (prob, ns, r, done)."""
    inner = bridge_env.unwrapped
    inner.P = [0.7, 0.2, 0.1]
    tm = inner.transition_matrix
    assert isinstance(tm, dict)
    assert set(tm.keys()) == set(range(inner.nS))
    # Every (s, a) entry should be a non-empty list of 4-tuples.
    for s in tm:
        for a in tm[s]:
            assert isinstance(tm[s][a], list)
            assert len(tm[s][a]) >= 1
            prob, next_state, reward, done = tm[s][a][0]
            assert 0.0 <= prob <= 1.0
            assert isinstance(next_state, (int, np.integer))
            assert isinstance(done, (bool, np.bool_))


def test_bridge_transition_matrix_uniform_uses_global_P(bridge_env):
    """In non-split mode, every cell's slip distribution should equal P."""
    inner = bridge_env.unwrapped
    inner.split_probs = False
    inner.P = [0.7, 0.2, 0.1]
    tm = inner.transition_matrix
    # Pick a non-terminal cell (row 2 is the bridge: G F F F S F F G).
    s_left = 2 * inner.ncol + 1   # row 2, col 1
    s_right = 2 * inner.ncol + 6  # row 2, col 6
    # First slip option uses transition_prob[0] = 0.7 on both sides.
    assert pytest.approx(tm[s_left][0][0][0], abs=1e-9) == 0.7
    assert pytest.approx(tm[s_right][0][0][0], abs=1e-9) == 0.7


def test_bridge_transition_matrix_split_uses_per_side_P(bridge_env):
    """In split mode, transition_matrix should pick the per-side slip."""
    inner = bridge_env.unwrapped
    inner.split_probs = True
    inner.P_left = [0.9, 0.05, 0.05]
    inner.P_right = [0.4, 0.3, 0.3]
    tm = inner.transition_matrix
    s_left = 2 * inner.ncol + 1
    s_right = 2 * inner.ncol + 6
    assert pytest.approx(tm[s_left][0][0][0], abs=1e-9) == 0.9
    assert pytest.approx(tm[s_right][0][0][0], abs=1e-9) == 0.4


def test_bridge_get_loc_based_prob_typo_alias(bridge_env):
    """The original method was misspelled ``get_loc_basedProb``. Both the
    fixed name and the legacy alias must work."""
    inner = bridge_env.unwrapped
    inner.P_left = [0.9, 0.05, 0.05]
    inner.P_right = [0.4, 0.3, 0.3]
    # left half: col < ncol // 2
    assert inner.get_loc_based_prob([2, 1]) == [0.9, 0.05, 0.05]
    assert inner.get_loc_based_prob([2, 6]) == [0.4, 0.3, 0.3]
    # legacy typo'd name still callable
    assert inner.get_loc_basedProb([2, 1]) == [0.9, 0.05, 0.05]


def test_bridge_action_to_state_mapping(bridge_env):
    """Action -> (row, col) delta mapping under deterministic P=[1,0,0]:
        LEFT(0)  -> ( 0,-1)
        DOWN(1)  -> (+1, 0)
        RIGHT(2) -> ( 0,+1)
        UP(3)    -> (-1, 0)
    Out-of-bound moves are no-ops (the agent stays put).
    """
    inner = bridge_env.unwrapped
    bridge_env.reset(seed=0)
    inner.P = [1.0, 0.0, 0.0]

    def state(r, c): return r * inner.ncol + c

    # (start_row, start_col, action, expected_row, expected_col)
    cases = [
        # From start S=(2,4) -- one step in each cardinal direction
        (2, 4, 0, 2, 3),   # LEFT
        (2, 4, 1, 3, 4),   # DOWN
        (2, 4, 2, 2, 5),   # RIGHT
        (2, 4, 3, 1, 4),   # UP
        # An interior frozen cell (1,1)
        (1, 1, 0, 1, 0),   # LEFT
        (1, 1, 1, 2, 1),   # DOWN
        (1, 1, 2, 1, 2),   # RIGHT
        # Out-of-bound: corners stay put
        (1, 0, 0, 1, 0),   # LEFT off the west edge
        (2, 7, 2, 2, 7),   # RIGHT off the east edge (this is also G, but
                           # we only test the coord mapping; terminating
                           # is fine since we don't read the next step)
    ]

    for sr, sc, a, er, ec in cases:
        inner.s = state(sr, sc)
        # Reset P after every step in case the parent NSWrapper mutated it
        inner.P = [1.0, 0.0, 0.0]
        obs, _, _, _, _ = bridge_env.step(a)
        s = obs["state"] if isinstance(obs, dict) else int(obs)
        assert int(s) == state(er, ec), (
            f"action={a} from ({sr},{sc}): expected ({er},{ec}) got "
            f"({divmod(int(s), inner.ncol)})"
        )


def test_bridge_slip_pattern(bridge_env):
    """With P=[0,1,0] every action is forced to slip to (a+1) % 4.
    From S=(2,4) the slipped destinations are:
        LEFT(0)  -> slip DOWN(1)  -> (3,4)
        DOWN(1)  -> slip RIGHT(2) -> (2,5)
        RIGHT(2) -> slip UP(3)    -> (1,4)
        UP(3)    -> slip LEFT(0)  -> (2,3)
    """
    inner = bridge_env.unwrapped
    bridge_env.reset(seed=0)

    expected = {0: (3, 4), 1: (2, 5), 2: (1, 4), 3: (2, 3)}
    for a, (er, ec) in expected.items():
        inner.s = 2 * inner.ncol + 4   # S
        inner.P = [0.0, 1.0, 0.0]      # always slip to (a+1)%4
        obs, _, _, _, _ = bridge_env.step(a)
        s = obs["state"] if isinstance(obs, dict) else int(obs)
        assert int(s) == er * inner.ncol + ec, (
            f"slip(+1) on action {a}: expected ({er},{ec})"
        )


def test_bridge_step_uses_selected_distribution(bridge_env):
    """When split_probs is on, step must sample from the side-local P,
    not the global P. (Regression for the original step() bug that
    always used ``self.P`` regardless of the selected distribution.)"""
    bridge_env.reset(seed=0)
    inner = bridge_env.unwrapped
    inner.split_probs = True
    # Deterministic on left, fully chaotic on right
    inner.P_left = [1.0, 0.0, 0.0]
    inner.P_right = [0.0, 1.0, 0.0]
    inner.P = [0.0, 0.0, 1.0]   # would crash if step ever used self.P here
    # Force agent into a left-side cell, take a step
    inner.s = 2 * inner.ncol + 1
    obs, r, term, trunc, info = bridge_env.step(0)
    assert info["prob"] == [1.0, 0.0, 0.0]
    # Now force into a right-side cell
    bridge_env.reset(seed=0)
    inner.split_probs = True
    inner.P_left = [1.0, 0.0, 0.0]
    inner.P_right = [0.0, 1.0, 0.0]
    inner.P = [0.0, 0.0, 1.0]
    inner.s = 2 * inner.ncol + 6
    obs, r, term, trunc, info = bridge_env.step(0)
    assert info["prob"] == [0.0, 1.0, 0.0]


def test_bridge_in_tunable_params_registry():
    """Bridge needs to be in TUNABLE_PARAMS or NSBridgeWrapper init asserts."""
    assert "Bridge" in TUNABLE_PARAMS
    bridge_keys = set(TUNABLE_PARAMS["Bridge"].keys())
    assert {"P", "P_left", "P_right"} <= bridge_keys


# ---------------------------------------------------------------------------
# NSBridgeWrapper -- uniform mode
# ---------------------------------------------------------------------------

def _no_update():
    return DistributionNoUpdate(ContinuousScheduler())


def _drift(rate=0.05):
    return UniformDrift(ContinuousScheduler(), rate=rate)


@pytest.fixture
def uniform_wrapped():
    env = gym.make("ns_gym/Bridge-v0")
    return NSBridgeWrapper(env, {"P": _no_update()},
                           initial_prob_dist=[0.8, 0.1, 0.1])


def test_uniform_init_sets_P(uniform_wrapped):
    assert uniform_wrapped.unwrapped.split_probs is False
    assert uniform_wrapped.unwrapped.P == [0.8, 0.1, 0.1]


def test_uniform_init_validates_tunable_params():
    """An unknown tunable param must trip the NSWrapper guard assertion."""
    env = gym.make("ns_gym/Bridge-v0")
    with pytest.raises(AssertionError):
        NSBridgeWrapper(env, {"not_a_real_param": _no_update()})


def test_uniform_step_returns_dict_obs(uniform_wrapped):
    obs, info = uniform_wrapped.reset(seed=0)
    assert isinstance(obs, dict)
    assert "state" in obs
    obs, r, term, trunc, info = uniform_wrapped.step(2)
    assert isinstance(obs, dict)
    assert info.get("prob") == [0.8, 0.1, 0.1]


def test_uniform_reset_restores_initial_P(uniform_wrapped):
    """A drifting wrapper must restore ``initial_prob_dist`` on reset."""
    env = gym.make("ns_gym/Bridge-v0")
    wrap = NSBridgeWrapper(env, {"P": _drift(rate=0.1)},
                           initial_prob_dist=[0.9, 0.05, 0.05])
    wrap.reset(seed=0)
    for _ in range(20):
        wrap.step(0)
    drifted = list(wrap.unwrapped.P)
    assert drifted != [0.9, 0.05, 0.05], "drift didn't move P"
    wrap.reset(seed=0)
    assert wrap.unwrapped.P == [0.9, 0.05, 0.05]


# ---------------------------------------------------------------------------
# NSBridgeWrapper -- split mode
# ---------------------------------------------------------------------------

@pytest.fixture
def split_wrapped():
    env = gym.make("ns_gym/Bridge-v0")
    return NSBridgeWrapper(
        env,
        {"P_left": _drift(rate=0.05), "P_right": _no_update()},
        initial_prob_dist=([0.9, 0.05, 0.05], [0.5, 0.25, 0.25]),
    )


def test_split_init_enables_split_probs(split_wrapped):
    assert split_wrapped.unwrapped.split_probs is True
    assert split_wrapped.unwrapped.P_left == [0.9, 0.05, 0.05]
    assert split_wrapped.unwrapped.P_right == [0.5, 0.25, 0.25]


def test_split_drift_independent(split_wrapped):
    """Left side should drift toward uniform; right side should not move."""
    split_wrapped.reset(seed=0)
    initial_left = list(split_wrapped.unwrapped.P_left)
    initial_right = list(split_wrapped.unwrapped.P_right)
    for _ in range(20):
        split_wrapped.step(0)
    assert split_wrapped.unwrapped.P_left != initial_left
    assert split_wrapped.unwrapped.P_right == initial_right


def test_split_reset_restores_both_sides(split_wrapped):
    split_wrapped.reset(seed=0)
    for _ in range(10):
        split_wrapped.step(0)
    split_wrapped.reset(seed=0)
    assert split_wrapped.unwrapped.P_left == [0.9, 0.05, 0.05]
    assert split_wrapped.unwrapped.P_right == [0.5, 0.25, 0.25]


def test_split_only_one_side_specified():
    """Specifying only P_left should still enable split mode and leave
    P_right at its initial value."""
    env = gym.make("ns_gym/Bridge-v0")
    wrap = NSBridgeWrapper(
        env,
        {"P_left": _drift(rate=0.1)},
        initial_prob_dist=([1.0, 0.0, 0.0], [0.6, 0.2, 0.2]),
    )
    wrap.reset(seed=0)
    assert wrap.unwrapped.split_probs is True
    initial_right = list(wrap.unwrapped.P_right)
    for _ in range(10):
        wrap.step(0)
    assert wrap.unwrapped.P_right == initial_right


def test_split_transition_matrix_reflects_per_side(split_wrapped):
    split_wrapped.reset(seed=0)
    tm = split_wrapped.unwrapped.transition_matrix
    inner = split_wrapped.unwrapped
    s_left = 2 * inner.ncol + 1
    s_right = 2 * inner.ncol + 6
    assert pytest.approx(tm[s_left][0][0][0], abs=1e-9) == 0.9
    assert pytest.approx(tm[s_right][0][0][0], abs=1e-9) == 0.5


# ---------------------------------------------------------------------------
# Deepcopy / planning-env tests
# ---------------------------------------------------------------------------

def test_uniform_deepcopy_is_isolated(uniform_wrapped):
    uniform_wrapped.reset(seed=0)
    sim = copy.deepcopy(uniform_wrapped)
    assert sim.is_sim_env
    assert sim.unwrapped.P == uniform_wrapped.unwrapped.P
    # Mutating the copy must not affect the original.
    sim.unwrapped.P = [0.0, 0.5, 0.5]
    assert uniform_wrapped.unwrapped.P == [0.8, 0.1, 0.1]


# ---------------------------------------------------------------------------
# P-snapshot semantics: env.unwrapped.P (and P_left/P_right) must be the
# CURRENT slip distribution at time t, never a precomputed future schedule.
# ---------------------------------------------------------------------------

def _drift_uniform_env():
    env = gym.make("ns_gym/Bridge-v0")
    return NSBridgeWrapper(env, {"P": _drift(rate=0.05)},
                           initial_prob_dist=[0.9, 0.05, 0.05])


def test_uniform_P_is_current_snapshot_not_future():
    """P read twice between steps must be identical; reads only update via step()."""
    env = _drift_uniform_env()
    env.reset(seed=0)
    P_a = list(env.unwrapped.P)
    P_b = list(env.unwrapped.P)
    P_c = list(env.unwrapped.P)
    assert P_a == P_b == P_c
    env.step(env.action_space.sample())
    P_d = list(env.unwrapped.P)
    assert P_d != P_a, "P must drift after step()"


def test_uniform_deepcopy_freezes_P_against_future_drift():
    """A deepcopy at time t must NOT inherit subsequent drift from the original."""
    env = _drift_uniform_env()
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    snap = copy.deepcopy(env)
    snap_P_at_capture = list(snap.unwrapped.P)
    for _ in range(10):
        env.step(env.action_space.sample())
    assert list(snap.unwrapped.P) == snap_P_at_capture, (
        "deepcopy leaked future drift from the original"
    )
    assert list(env.unwrapped.P) != snap_P_at_capture, (
        "original env did not drift after the snapshot"
    )


def test_split_P_left_right_deepcopy_freezes_against_future():
    """Same snapshot guarantee on the per-side distributions in split mode."""
    env = gym.make("ns_gym/Bridge-v0")
    ns_env = NSBridgeWrapper(
        env,
        {"P_left": _drift(rate=0.05), "P_right": _drift(rate=0.02)},
        initial_prob_dist=([0.9, 0.05, 0.05], [0.7, 0.15, 0.15]),
    )
    ns_env.reset(seed=0)
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
    snap = copy.deepcopy(ns_env)
    L_snap = list(snap.unwrapped.P_left)
    R_snap = list(snap.unwrapped.P_right)
    for _ in range(10):
        ns_env.step(ns_env.action_space.sample())
    assert list(snap.unwrapped.P_left) == L_snap
    assert list(snap.unwrapped.P_right) == R_snap
    assert list(ns_env.unwrapped.P_left) != L_snap
    assert list(ns_env.unwrapped.P_right) != R_snap


def test_planning_env_freezes_P_no_step_crash():
    """get_planning_env() returns an env whose P stays frozen across step().

    Regression for: NSBridgeWrapper.step used to pass env_change=None when
    is_sim_env=True, but the parent NSWrapper.step expected dicts -- so any
    planner that called step() on a planning env crashed with
    AttributeError: 'NoneType' object has no attribute 'items'.
    """
    env = _drift_uniform_env()
    env.reset(seed=0)
    for _ in range(10):
        env.step(env.action_space.sample())
    pe = env.get_planning_env()
    P_before = list(pe.unwrapped.P)
    # The actual regression -- this used to raise AttributeError.
    for _ in range(20):
        obs, _, term, trunc, _ = pe.step(pe.action_space.sample())
        if term or trunc:
            pe.reset()
    assert list(pe.unwrapped.P) == P_before, (
        "planning_env P drifted; it must stay frozen at the captured snapshot"
    )


def test_split_deepcopy_preserves_both_sides(split_wrapped):
    split_wrapped.reset(seed=0)
    sim = copy.deepcopy(split_wrapped)
    assert sim.is_sim_env
    assert sim.unwrapped.split_probs is True
    assert sim.unwrapped.P_left == split_wrapped.unwrapped.P_left
    assert sim.unwrapped.P_right == split_wrapped.unwrapped.P_right
    # Isolation
    sim.unwrapped.P_left = [0.0, 0.5, 0.5]
    assert split_wrapped.unwrapped.P_left == [0.9, 0.05, 0.05]
