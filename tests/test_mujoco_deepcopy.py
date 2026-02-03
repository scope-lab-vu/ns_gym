import pytest
import gymnasium as gym
import numpy as np
import copy
from copy import deepcopy

from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate
from ns_gym.wrappers.mujoco_env import MujocoWrapper
from ns_gym.base import Reward, TUNABLE_PARAMS


# --- Constants ---

MUJOCO_ENV_IDS = [
    "Ant-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Reacher-v5",
    "Swimmer-v5",
    "Pusher-v5",
]


# --- Fixtures ---

@pytest.fixture
def mj_params():
    """Tunable params for each MuJoCo env (representative subset)."""
    fn = IncrementUpdate(ContinuousScheduler(), k=0.1)
    return {
        "Ant-v5": {"torso_mass": fn},
        "HalfCheetah-v5": {"torso_mass": fn, "floor_friction": fn},
        "Hopper-v5": {"torso_mass": fn, "thigh_mass": fn},
        "Humanoid-v5": {"torso_mass": fn},
        "HumanoidStandup-v5": {"torso_mass": fn},
        "InvertedPendulum-v5": {"pole_mass": fn, "cart_mass": fn},
        "InvertedDoublePendulum-v5": {"cart_mass": fn, "pole1_mass": fn},
        "Reacher-v5": {"body0_mass": fn, "body1_mass": fn},
        "Swimmer-v5": {"body_mid_mass": fn},
        "Pusher-v5": {"r_shoulder_pan_link_mass": fn},
    }


# --- Helpers ---

def _make_mj_env(env_id, mj_params, **wrapper_kwargs):
    env = gym.make(env_id)
    return MujocoWrapper(env, mj_params[env_id], **wrapper_kwargs)


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

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_deepcopy_sets_is_sim_env(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)

    assert sim_env.is_sim_env is True
    assert ns_env.is_sim_env is False
    assert isinstance(sim_env, MujocoWrapper)


# ============================================================
# Deepcopy: params match at copy point
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_deepcopy_params_match(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params)
    ns_env.reset(seed=42)
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)

    for param in mj_params[env_id]:
        orig_val = ns_env._get_param_value(param)
        copy_val = sim_env._get_param_value(param)
        assert np.allclose(orig_val, copy_val), (
            f"Param '{param}' mismatch: original={orig_val}, copy={copy_val}"
        )


# ============================================================
# Deepcopy: state (qpos, qvel) matches at copy point
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_deepcopy_state_matches(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)

    assert np.allclose(
        sim_env.unwrapped.data.qpos, ns_env.unwrapped.data.qpos
    ), "qpos mismatch after deepcopy"
    assert np.allclose(
        sim_env.unwrapped.data.qvel, ns_env.unwrapped.data.qvel
    ), "qvel mismatch after deepcopy"


# ============================================================
# Deepcopy: independence (stepping copy doesn't affect original)
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_deepcopy_independence(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params, in_sim_change=True)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    orig_params = {}
    for param in mj_params[env_id]:
        orig_params[param] = deepcopy(ns_env._get_param_value(param))

    sim_env = copy.deepcopy(ns_env)

    for _ in range(10):
        action = sim_env.action_space.sample()
        obs, _, done, trunc, _ = sim_env.step(action)
        if done or trunc:
            sim_env.reset(seed=42)

    for param in mj_params[env_id]:
        assert np.allclose(ns_env._get_param_value(param), orig_params[param]), (
            f"Original env param '{param}' was mutated by stepping the copy"
        )


# ============================================================
# Deepcopy: sim_env freezes params when in_sim_change=False
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_deepcopy_sim_env_freezes_params(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params, in_sim_change=False)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    assert sim_env.is_sim_env is True

    frozen_params = {}
    for param in mj_params[env_id]:
        frozen_params[param] = deepcopy(sim_env._get_param_value(param))

    for _ in range(5):
        action = sim_env.action_space.sample()
        obs, _, done, trunc, _ = sim_env.step(action)
        if done or trunc:
            break

    for param in mj_params[env_id]:
        assert np.allclose(sim_env._get_param_value(param), frozen_params[param]), (
            f"Sim env param '{param}' changed despite in_sim_change=False"
        )


# ============================================================
# Deepcopy: sim_env updates params when in_sim_change=True
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_deepcopy_sim_env_updates_params(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params, in_sim_change=True)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    assert sim_env.is_sim_env is True

    initial_params = {}
    for param in mj_params[env_id]:
        initial_params[param] = deepcopy(sim_env._get_param_value(param))

    for _ in range(5):
        action = sim_env.action_space.sample()
        obs, _, done, trunc, _ = sim_env.step(action)
        if done or trunc:
            break

    any_changed = False
    for param in mj_params[env_id]:
        if not np.allclose(sim_env._get_param_value(param), initial_params[param]):
            any_changed = True
    assert any_changed, "No params changed on sim_env despite in_sim_change=True"


# ============================================================
# Deepcopy: t value preserved
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_deepcopy_t_value_preserved(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params)
    ns_env.reset(seed=42)
    for _ in range(7):
        ns_env.step(ns_env.action_space.sample())

    expected_t = ns_env.t
    sim_env = copy.deepcopy(ns_env)
    assert sim_env.t == expected_t


# ============================================================
# Deepcopy: at various episode points
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
@pytest.mark.parametrize("copy_at_step", [0, 1, 5, 10])
def test_deepcopy_at_various_points(mj_params, env_id, copy_at_step):
    ns_env = _make_mj_env(env_id, mj_params)
    ns_env.reset(seed=42)
    for _ in range(copy_at_step):
        obs, _, done, trunc, _ = ns_env.step(ns_env.action_space.sample())
        if done or trunc:
            ns_env.reset(seed=42)

    sim_env = copy.deepcopy(ns_env)

    assert sim_env.is_sim_env is True
    assert sim_env.t == ns_env.t
    assert isinstance(sim_env, MujocoWrapper)


# ============================================================
# Deepcopy: scalar_reward flag propagated
# ============================================================

@pytest.mark.parametrize("env_id", ["InvertedPendulum-v5", "HalfCheetah-v5"])
@pytest.mark.parametrize("scalar_reward", [True, False])
def test_deepcopy_scalar_reward_propagated(mj_params, env_id, scalar_reward):
    ns_env = _make_mj_env(env_id, mj_params, scalar_reward=scalar_reward)
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

@pytest.mark.parametrize("env_id", ["Ant-v5", "InvertedPendulum-v5"])
def test_deepcopy_notification_flags_preserved(mj_params, env_id):
    ns_env = _make_mj_env(
        env_id, mj_params,
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

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_deepcopy_wrapper_stack_intact(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)
    _validate_no_double_wrap(sim_env, MujocoWrapper)


# ============================================================
# Deepcopy: stationary copy produces identical trajectories
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_deepcopy_stationary_trajectories_match(env_id):
    """With k=0 (no param changes), a deepcopy stepped with the same actions
    must produce identical qpos/qvel at every step."""
    stationary_fn = IncrementUpdate(ContinuousScheduler(), k=0.0)
    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__

    # Use a single representative param per env with zero increment
    first_param = list(TUNABLE_PARAMS[env_name].keys())[0]
    params = {first_param: stationary_fn}

    ns_env = MujocoWrapper(env, params, in_sim_change=True)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    sim_env = copy.deepcopy(ns_env)

    # Generate a fixed action sequence
    np.random.seed(123)
    actions = [ns_env.action_space.sample() for _ in range(20)]

    for i, action in enumerate(actions):
        obs_orig, _, done_orig, trunc_orig, _ = ns_env.step(action)
        obs_sim, _, done_sim, trunc_sim, _ = sim_env.step(action)

        assert np.allclose(
            ns_env.unwrapped.data.qpos, sim_env.unwrapped.data.qpos
        ), f"qpos diverged at step {i}"
        assert np.allclose(
            ns_env.unwrapped.data.qvel, sim_env.unwrapped.data.qvel
        ), f"qvel diverged at step {i}"

        if done_orig or trunc_orig or done_sim or trunc_sim:
            break


# ============================================================
# get_planning_env: returns correct type
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_get_planning_env_returns_wrapper(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    planning_env = ns_env.get_planning_env()
    assert isinstance(planning_env, MujocoWrapper)
    assert planning_env.is_sim_env is True


# ============================================================
# get_planning_env: requires reset
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_get_planning_env_requires_reset(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params)
    with pytest.raises(AssertionError, match="must be reset"):
        ns_env.get_planning_env()


# ============================================================
# get_planning_env: delta_change_notification=False
#   -> state matches, params reset to initial
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_get_planning_env_delta_notification_false(mj_params, env_id):
    """When delta_change_notification=False the planning env should have the
    same simulation state (qpos/qvel) but tunable parameters restored to
    their initial values."""
    ns_env = _make_mj_env(
        env_id, mj_params,
        change_notification=False,
        delta_change_notification=False,
    )
    ns_env.reset(seed=42)
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())

    # Confirm params have actually drifted from initial before calling
    for param in mj_params[env_id]:
        current_val = ns_env._get_param_value(param)
        initial_val = ns_env.initial_params[param]
        assert not np.allclose(current_val, initial_val), (
            f"Param '{param}' should have changed after stepping, "
            "otherwise this test is not exercising the reset path"
        )

    planning_env = ns_env.get_planning_env()

    # State must match
    assert np.allclose(
        planning_env.unwrapped.data.qpos, ns_env.unwrapped.data.qpos
    ), "Planning env qpos should match original"
    assert np.allclose(
        planning_env.unwrapped.data.qvel, ns_env.unwrapped.data.qvel
    ), "Planning env qvel should match original"

    # Params must be reset to initial values
    for param in mj_params[env_id]:
        planning_val = planning_env._get_param_value(param)
        initial_val = ns_env.initial_params[param]
        assert np.allclose(planning_val, initial_val), (
            f"Planning env param '{param}' should be reset to initial value. "
            f"Got {planning_val}, expected {initial_val}"
        )


# ============================================================
# get_planning_env: delta_change_notification=True
#   -> state matches, params match current values
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_get_planning_env_delta_notification_true(mj_params, env_id):
    """When delta_change_notification=True the planning env should have the
    same simulation state (qpos/qvel) AND the same current tunable parameter
    values as the original."""
    ns_env = _make_mj_env(
        env_id, mj_params,
        change_notification=True,
        delta_change_notification=True,
    )
    ns_env.reset(seed=42)
    for _ in range(5):
        ns_env.step(ns_env.action_space.sample())

    planning_env = ns_env.get_planning_env()

    # State must match
    assert np.allclose(
        planning_env.unwrapped.data.qpos, ns_env.unwrapped.data.qpos
    ), "Planning env qpos should match original"
    assert np.allclose(
        planning_env.unwrapped.data.qvel, ns_env.unwrapped.data.qvel
    ), "Planning env qvel should match original"

    # Params must match current (drifted) values
    for param in mj_params[env_id]:
        planning_val = planning_env._get_param_value(param)
        current_val = ns_env._get_param_value(param)
        assert np.allclose(planning_val, current_val), (
            f"Planning env param '{param}' should match current value. "
            f"Got {planning_val}, expected {current_val}"
        )


# ============================================================
# get_planning_env: independence from original
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_get_planning_env_independence(mj_params, env_id):
    ns_env = _make_mj_env(env_id, mj_params, in_sim_change=True)
    ns_env.reset(seed=42)
    for _ in range(3):
        ns_env.step(ns_env.action_space.sample())

    orig_params = {}
    for param in mj_params[env_id]:
        orig_params[param] = deepcopy(ns_env._get_param_value(param))
    orig_qpos = ns_env.unwrapped.data.qpos.copy()

    planning_env = ns_env.get_planning_env()
    for _ in range(10):
        action = planning_env.action_space.sample()
        _, _, done, trunc, _ = planning_env.step(action)
        if done or trunc:
            planning_env.reset(seed=42)

    for param in mj_params[env_id]:
        assert np.allclose(ns_env._get_param_value(param), orig_params[param]), (
            f"Original param '{param}' was mutated by stepping planning env"
        )
    assert np.allclose(ns_env.unwrapped.data.qpos, orig_qpos), (
        "Original qpos was mutated by stepping planning env"
    )
