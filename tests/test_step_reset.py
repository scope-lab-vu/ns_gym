import pytest
import gymnasium as gym
import numpy as np
from copy import deepcopy

import ns_gym.base as base
from ns_gym.base import TUNABLE_PARAMS, Reward
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate, DistributionIncrementUpdate, RandomWalk
from ns_gym.wrappers import NSClassicControlWrapper, NSCliffWalkingWrapper, NSFrozenLakeWrapper
from ns_gym.wrappers.mujoco_env import MujocoWrapper


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


# ============================================================
# Issue 1: MuJoCo double time increment regression test
# ============================================================

MUJOCO_TIME_TEST_IDS = [
    "InvertedPendulum-v5",
    "HalfCheetah-v5",
]


@pytest.mark.parametrize("env_id", MUJOCO_TIME_TEST_IDS)
def test_step_increments_t_mujoco(env_id):
    """MuJoCo wrapper should increment t by 1 per step, not 2."""
    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    first_param = list(TUNABLE_PARAMS[env_name].keys())[0]
    fn = IncrementUpdate(ContinuousScheduler(), k=0.1)
    ns_env = MujocoWrapper(env, {first_param: fn})
    ns_env.reset(seed=42)

    for step_num in range(1, 6):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            break
        assert ns_env.t == step_num, (
            f"Expected t={step_num}, got t={ns_env.t} (double increment bug?)"
        )
        assert obs["relative_time"] == step_num

    ns_env.close()


# ============================================================
# Issue 2: persistent_params tests
# ============================================================

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
    from ns_gym.wrappers.mujoco_env import param_look_up

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
# Issue 3: Seeding tests — Classic Control
# ============================================================

CC_SEEDING_PARAMS = {
    "CartPole-v1": "masspole",
    "Acrobot-v1": "LINK_LENGTH_1",
    "MountainCar-v0": "force",
    "MountainCarContinuous-v0": "power",
    "Pendulum-v1": "m",
}


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_reset_no_seed_different_rng_sequence_classic_control(env_id):
    """reset() with no seed should produce different RNG sequences each episode."""
    param = CC_SEEDING_PARAMS[env_id]
    env = gym.make(env_id)
    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.01, seed=42)
    ns_env = NSClassicControlWrapper(env, {param: fn})

    # First episode with explicit seed
    ns_env.reset(seed=0)
    trajectory_a = []
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
        trajectory_a.append(getattr(env.unwrapped, param))

    # Second episode without seed — RNG should continue
    ns_env.reset()
    trajectory_b = []
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
        trajectory_b.append(getattr(env.unwrapped, param))

    assert trajectory_a != trajectory_b, (
        f"[{env_id}] reset() with no seed should produce different RNG sequences"
    )


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_reset_same_seed_reproducible_classic_control(env_id):
    """reset(seed=X) should produce identical RNG sequences each time."""
    param = CC_SEEDING_PARAMS[env_id]
    env = gym.make(env_id)
    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.01, seed=42)
    ns_env = NSClassicControlWrapper(env, {param: fn})

    ns_env.reset(seed=0)
    trajectory_a = []
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
        trajectory_a.append(getattr(env.unwrapped, param))

    ns_env.reset(seed=0)
    trajectory_b = []
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
        trajectory_b.append(getattr(env.unwrapped, param))

    assert trajectory_a == trajectory_b, (
        f"[{env_id}] reset(seed=0) twice should produce identical RNG sequences"
    )


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_reset_no_seed_multiple_episodes_differ_classic_control(env_id):
    """Multiple reset() calls without seed should each produce different trajectories."""
    param = CC_SEEDING_PARAMS[env_id]
    env = gym.make(env_id)
    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.01, seed=42)
    ns_env = NSClassicControlWrapper(env, {param: fn})

    ns_env.reset(seed=0)  # Initial seed

    trajectories = []
    for _ in range(3):
        ns_env.reset()  # No seed
        traj = []
        for _ in range(5):
            ns_env.step(ns_env.action_space.sample())
            traj.append(getattr(env.unwrapped, param))
        trajectories.append(traj)

    # All 3 trajectories should be different from each other
    assert trajectories[0] != trajectories[1], (
        f"[{env_id}] Episodes 1 and 2 should differ without explicit seed"
    )
    assert trajectories[1] != trajectories[2], (
        f"[{env_id}] Episodes 2 and 3 should differ without explicit seed"
    )


# ============================================================
# Issue 3: Seeding tests — MuJoCo
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_TIME_TEST_IDS)
def test_reset_no_seed_different_rng_sequence_mujoco(env_id):
    """MuJoCo: reset() with no seed should produce different RNG sequences."""
    from ns_gym.wrappers.mujoco_env import param_look_up

    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    params = list(TUNABLE_PARAMS[env_name].keys())
    param_name = next((p for p in params if p != "gravity"), params[0])

    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.01, seed=42)
    ns_env = MujocoWrapper(env, {param_name: fn})
    getter, _ = param_look_up(env_name, param_name)[0]

    ns_env.reset(seed=0)
    trajectory_a = []
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
        trajectory_a.append(float(getter(env.unwrapped)))

    ns_env.reset()
    trajectory_b = []
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
        trajectory_b.append(float(getter(env.unwrapped)))

    assert trajectory_a != trajectory_b, (
        f"[{env_id}] reset() with no seed should produce different RNG sequences"
    )
    ns_env.close()


@pytest.mark.parametrize("env_id", MUJOCO_TIME_TEST_IDS)
def test_reset_same_seed_reproducible_mujoco(env_id):
    """MuJoCo: reset(seed=X) should produce identical RNG sequences."""
    from ns_gym.wrappers.mujoco_env import param_look_up

    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    params = list(TUNABLE_PARAMS[env_name].keys())
    param_name = next((p for p in params if p != "gravity"), params[0])

    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.01, seed=42)
    ns_env = MujocoWrapper(env, {param_name: fn})
    getter, _ = param_look_up(env_name, param_name)[0]

    ns_env.reset(seed=0)
    trajectory_a = []
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
        trajectory_a.append(float(getter(env.unwrapped)))

    ns_env.reset(seed=0)
    trajectory_b = []
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
        trajectory_b.append(float(getter(env.unwrapped)))

    assert trajectory_a == trajectory_b, (
        f"[{env_id}] reset(seed=0) twice should produce identical RNG sequences"
    )
    ns_env.close()


# ============================================================
# Issue 3: Seeding tests — Gridworld
# ============================================================

@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_reset_no_seed_different_rng_sequence_gridworld(gw_wrappers, env_id):
    """Gridworld: reset() with no seed should produce different RNG sequences."""
    from ns_gym.update_functions import RandomCategorical

    env = gym.make(env_id)
    WrapperClass = gw_wrappers[env_id]
    fn = RandomCategorical(ContinuousScheduler(), seed=42)
    ns_env = WrapperClass(env, {"P": fn})

    ns_env.reset(seed=0)
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
    tp_a = ns_env.transition_prob[:]

    ns_env.reset()
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
    tp_b = ns_env.transition_prob[:]

    assert tp_a != tp_b, (
        f"[{env_id}] reset() with no seed should produce different transition probs"
    )


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_reset_same_seed_reproducible_gridworld(gw_wrappers, env_id):
    """Gridworld: reset(seed=X) should produce identical RNG sequences."""
    from ns_gym.update_functions import RandomCategorical

    env = gym.make(env_id)
    WrapperClass = gw_wrappers[env_id]
    fn = RandomCategorical(ContinuousScheduler(), seed=42)
    ns_env = WrapperClass(env, {"P": fn})

    ns_env.reset(seed=0)
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
    tp_a = ns_env.transition_prob[:]

    ns_env.reset(seed=0)
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())
    tp_b = ns_env.transition_prob[:]

    assert tp_a == tp_b, (
        f"[{env_id}] reset(seed=0) twice should produce identical transition probs"
    )


# ============================================================
# Issue 3: Seeding + persistent_params interaction
# ============================================================

def test_persistent_params_seed_overrides_rng():
    """With persistent_params=True, explicit seed should still re-seed RNGs."""
    env = gym.make("CartPole-v1")
    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.01, seed=42)
    ns_env = NSClassicControlWrapper(env, {"masspole": fn}, persistent_params=True)

    # Track deltas (increments) rather than absolute values,
    # since persistent_params means the base value differs between episodes
    ns_env.reset(seed=0)
    deltas_a = []
    prev = env.unwrapped.masspole
    for _ in range(5):
        ns_env.step(0)
        cur = env.unwrapped.masspole
        deltas_a.append(cur - prev)
        prev = cur

    # Explicit seed should re-seed even with persistent_params
    ns_env.reset(seed=0)
    deltas_b = []
    prev = env.unwrapped.masspole
    for _ in range(5):
        ns_env.step(0)
        cur = env.unwrapped.masspole
        deltas_b.append(cur - prev)
        prev = cur

    assert np.allclose(deltas_a, deltas_b), (
        "With persistent_params=True, reset(seed=0) should produce same RNG-driven deltas"
    )


# ============================================================
# Issue 3: Planning env RNG divergence tests
# ============================================================

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


@pytest.mark.parametrize("env_id", MUJOCO_TIME_TEST_IDS)
def test_planning_env_rng_diverges_mujoco(env_id):
    """MuJoCo planning env should NOT predict future stochastic parameter changes."""
    from ns_gym.wrappers.mujoco_env import param_look_up

    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    params = list(TUNABLE_PARAMS[env_name].keys())
    param_name = next((p for p in params if p != "gravity"), params[0])

    fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.1, seed=42)
    ns_env = MujocoWrapper(
        env, {param_name: fn},
        change_notification=True,
        delta_change_notification=True,
    )
    getter, _ = param_look_up(env_name, param_name)[0]

    ns_env.reset(seed=0)

    planning_env = ns_env.get_planning_env()

    # --- Explicit RNG state comparison ---
    real_fn = ns_env.tunable_params[param_name]
    plan_fn = planning_env.tunable_params[param_name]
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
    ns_env = MujocoWrapper(
        env, {param_name: fn},
        change_notification=True,
        delta_change_notification=True,
    )
    getter, _ = param_look_up(env_name, param_name)[0]
    ns_env.reset(seed=0)

    planning_env = ns_env.get_planning_env()

    # Generate a fixed action sequence
    actions = [ns_env.action_space.sample() for _ in range(10)]

    real_trajectory = []
    planning_trajectory = []
    for action in actions:
        obs_real, _, done, trunc, _ = ns_env.step(action)
        if done or trunc:
            break
        real_trajectory.append(float(getter(env.unwrapped)))

        obs_plan, _, _, _, _ = planning_env.step(action)
        planning_trajectory.append(float(getter(planning_env.unwrapped)))

    if len(real_trajectory) > 0:
        assert real_trajectory != planning_trajectory, (
            f"[{env_id}] Planning env should have independent RNG — trajectories must diverge"
        )

    ns_env.close()


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_planning_env_rng_diverges_gridworld(gw_wrappers, env_id):
    """Gridworld planning env should NOT predict future stochastic parameter changes."""
    from ns_gym.update_functions import RandomCategorical

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
# Issue 3: Two independent envs with same seed produce identical results
# ============================================================

@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_two_envs_same_seed_identical_classic_control(env_id):
    """Two independently created envs with the same seed should produce identical
    state transitions and parameter updates."""
    param = CC_SEEDING_PARAMS[env_id]

    def run_episode(seed):
        env = gym.make(env_id)
        fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.1, seed=99)
        ns_env = NSClassicControlWrapper(env, {param: fn})
        ns_env.reset(seed=seed)

        states = []
        param_values = []
        action = 0 if hasattr(ns_env.action_space, 'n') else np.zeros(ns_env.action_space.shape)
        for _ in range(10):
            obs, _, done, trunc, _ = ns_env.step(action)
            if done or trunc:
                break
            states.append(obs["state"].tolist() if hasattr(obs["state"], "tolist") else obs["state"])
            param_values.append(float(getattr(env.unwrapped, param)))
        return states, param_values

    states_a, params_a = run_episode(seed=42)
    states_b, params_b = run_episode(seed=42)

    assert len(states_a) > 0, f"[{env_id}] No steps completed"
    assert params_a == params_b, (
        f"[{env_id}] Two envs with same seed should produce identical parameter updates"
    )
    assert states_a == states_b, (
        f"[{env_id}] Two envs with same seed should produce identical state transitions"
    )


@pytest.mark.parametrize("env_id", MUJOCO_TIME_TEST_IDS)
def test_two_envs_same_seed_identical_mujoco(env_id):
    """Two independently created MuJoCo envs with the same seed should produce
    identical state transitions and parameter updates."""
    from ns_gym.wrappers.mujoco_env import param_look_up

    env_name = gym.make(env_id).unwrapped.__class__.__name__
    params = list(TUNABLE_PARAMS[env_name].keys())
    param_name = next((p for p in params if p != "gravity"), params[0])

    # Pre-generate a fixed action sequence using a separate RNG
    action_rng = np.random.default_rng(seed=123)
    tmp_env = gym.make(env_id)
    actions = [action_rng.uniform(
        low=tmp_env.action_space.low,
        high=tmp_env.action_space.high,
    ).astype(tmp_env.action_space.dtype) for _ in range(10)]
    tmp_env.close()

    def run_episode(seed):
        env = gym.make(env_id)
        fn = RandomWalk(ContinuousScheduler(), mu=0, sigma=0.1, seed=99)
        ns_env = MujocoWrapper(env, {param_name: fn})
        getter, _ = param_look_up(env_name, param_name)[0]
        ns_env.reset(seed=seed)

        states = []
        param_values = []
        for action in actions:
            obs, _, done, trunc, _ = ns_env.step(action)
            if done or trunc:
                break
            states.append(obs["state"].tolist())
            param_values.append(float(getter(env.unwrapped)))
        ns_env.close()
        return states, param_values

    states_a, params_a = run_episode(seed=42)
    states_b, params_b = run_episode(seed=42)

    assert len(states_a) > 0, f"[{env_id}] No steps completed"
    assert params_a == params_b, (
        f"[{env_id}] Two envs with same seed should produce identical parameter updates"
    )
    assert states_a == states_b, (
        f"[{env_id}] Two envs with same seed should produce identical state transitions"
    )


@pytest.mark.parametrize("env_id", GRIDWORLD_ENV_IDS)
def test_two_envs_same_seed_identical_gridworld(gw_wrappers, env_id):
    """Two independently created gridworld envs with the same seed should produce
    identical state transitions and parameter updates."""
    from ns_gym.update_functions import RandomCategorical

    WrapperClass = gw_wrappers[env_id]

    # Pre-generate a fixed action sequence using a separate RNG
    action_rng = np.random.default_rng(seed=123)
    tmp_env = gym.make(env_id)
    actions = [int(action_rng.integers(tmp_env.action_space.n)) for _ in range(10)]
    tmp_env.close()

    def run_episode(seed):
        env = gym.make(env_id)
        fn = RandomCategorical(ContinuousScheduler(), seed=99)
        ns_env = WrapperClass(env, {"P": fn})
        ns_env.reset(seed=seed)

        states = []
        transition_probs = []
        for action in actions:
            obs, _, done, trunc, _ = ns_env.step(action)
            if done or trunc:
                break
            states.append(obs["state"])
            transition_probs.append(ns_env.transition_prob[:])
        return states, transition_probs

    states_a, tp_a = run_episode(seed=42)
    states_b, tp_b = run_episode(seed=42)

    assert len(tp_a) > 0, f"[{env_id}] No steps completed"
    assert tp_a == tp_b, (
        f"[{env_id}] Two envs with same seed should produce identical transition probs"
    )
    assert states_a == states_b, (
        f"[{env_id}] Two envs with same seed should produce identical state transitions"
    )


# ============================================================
# MuJoCo: state preserved after parameter update (mj_setConst fix)
# ============================================================

MUJOCO_STATE_CONSISTENCY_IDS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
]


@pytest.mark.parametrize("env_id", MUJOCO_STATE_CONSISTENCY_IDS)
def test_mujoco_state_preserved_after_param_update(env_id):
    """After a parameter update, qpos/qvel should reflect physics, not be reset.

    The dependency resolver calls mj_setConst (which recomputes derived model
    constants like body_subtreemass) but must preserve the integration state
    (qpos, qvel, time). This test verifies that stepping with parameter changes
    produces continuous state evolution, not state resets.
    """
    import mujoco
    from ns_gym.wrappers.mujoco_env import param_look_up

    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    params = list(TUNABLE_PARAMS[env_name].keys())
    param_name = next((p for p in params if p != "gravity"), params[0])

    fn = IncrementUpdate(ContinuousScheduler(start=0), k=0.5)
    ns_env = MujocoWrapper(env, {param_name: fn})
    ns_env.reset(seed=42)

    model = env.unwrapped.model
    data = env.unwrapped.data

    # Step once to get non-trivial state
    action = ns_env.action_space.sample()
    ns_env.step(action)

    qpos_after_step = data.qpos.copy()
    qvel_after_step = data.qvel.copy()
    time_after_step = data.time

    # qpos should NOT be all zeros (i.e. not reset to qpos0)
    assert not np.allclose(qpos_after_step, model.qpos0, atol=1e-10), (
        f"[{env_id}] qpos should not be at qpos0 after stepping — "
        "mj_setConst may have wiped the state"
    )

    # Step again — state should continue evolving, not snap back
    ns_env.step(action)

    qpos_after_second_step = data.qpos.copy()
    time_after_second_step = data.time

    assert time_after_second_step > time_after_step, (
        f"[{env_id}] Simulation time should advance after step"
    )
    assert not np.array_equal(qpos_after_second_step, qpos_after_step), (
        f"[{env_id}] qpos should change between steps"
    )
    assert not np.allclose(qpos_after_second_step, model.qpos0, atol=1e-10), (
        f"[{env_id}] qpos should not snap back to qpos0 after second step"
    )

    ns_env.close()


@pytest.mark.parametrize("env_id", MUJOCO_STATE_CONSISTENCY_IDS)
def test_mujoco_subtreemass_updated_after_mass_change(env_id):
    """After changing a mass parameter, body_subtreemass should be recomputed."""
    import mujoco
    from ns_gym.wrappers.mujoco_env import param_look_up

    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    params = list(TUNABLE_PARAMS[env_name].keys())
    # Pick a mass parameter
    mass_param = next((p for p in params if "mass" in p), None)
    if mass_param is None:
        pytest.skip(f"No mass parameter for {env_id}")

    fn = IncrementUpdate(ContinuousScheduler(start=0), k=1.0)
    ns_env = MujocoWrapper(env, {mass_param: fn})
    ns_env.reset(seed=42)

    model = env.unwrapped.model
    subtreemass_before = model.body_subtreemass.copy()

    # Step to trigger the parameter update + dependency resolver
    ns_env.step(ns_env.action_space.sample())

    subtreemass_after = model.body_subtreemass.copy()

    assert not np.array_equal(subtreemass_before, subtreemass_after), (
        f"[{env_id}] body_subtreemass should update after mass change — "
        "mj_setConst may not be called in _dependency_resolver"
    )

    ns_env.close()
