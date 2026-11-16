"""Behaviour specific to ``MujocoWrapper``: which parameters it accepts, that
each one is applied, restored on reset, and actually changes the simulation."""

import gymnasium as gym
import numpy as np
import pytest

from ns_gym import base
from ns_gym.base import TUNABLE_PARAMS
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate, NoUpdate
from ns_gym.wrappers import MujocoWrapper
from ns_gym.wrappers.mujoco_env import param_look_up

from _helpers import (
    MUJOCO_ENV_CLASS,
    MUJOCO_ENV_IDS,
    MUJOCO_ENV_PARAMS,
    fixed_actions,
    known_bug,
)

ENV_PARAM_IDS = [f"{env_id}-{param}" for env_id, param in MUJOCO_ENV_PARAMS]

HUMANOID_IDS = ["Humanoid-v5", "HumanoidStandup-v5"]


def _all_params_env(env_id, k=0.1, **wrapper_kwargs):
    """The env wrapped with every tunable parameter incrementing by ``k`` per step."""
    env = gym.make(env_id)
    params = TUNABLE_PARAMS[MUJOCO_ENV_CLASS[env_id]]
    param_map = {p: IncrementUpdate(ContinuousScheduler(), k=k) for p in params}
    return MujocoWrapper(env, param_map, **wrapper_kwargs)


# ============================================================
# Accepted parameters
# ============================================================

@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_env_class_names(env_id):
    """The id -> class-name map the tests parametrize from matches gymnasium."""
    assert gym.make(env_id).unwrapped.__class__.__name__ == MUJOCO_ENV_CLASS[env_id]


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_mujoco_wrapper_init(env_id):
    """The wrapper accepts every parameter in the allow-list at once."""
    ns_env = _all_params_env(env_id, change_notification=True, delta_change_notification=True)
    assert isinstance(ns_env, MujocoWrapper)


@pytest.mark.parametrize("env_id,param", MUJOCO_ENV_PARAMS, ids=ENV_PARAM_IDS)
def test_attribute_accessibility(env_id, param):
    """Every allowed parameter has a callable getter and setter."""
    for getter, setter in param_look_up(MUJOCO_ENV_CLASS[env_id], param):
        assert callable(getter), f"Getter for {param} in {env_id} is not callable"
        assert callable(setter), f"Setter for {param} in {env_id} is not callable"


def _settable_params(cls_name):
    """Parameter names ``param_look_up`` has an accessor for."""
    candidates = {name for table in base.MUJOCO_GETTERS.values() for name in table}
    candidates |= {"floor_friction", "rail_friction", "ground_friction", "geom_floor_friction", "gravity"}
    settable = set()
    for name in candidates:
        try:
            param_look_up(cls_name, name)
        except ValueError:
            continue
        settable.add(name)
    return settable


@pytest.mark.parametrize(
    "env_id",
    [
        pytest.param(
            env_id,
            marks=[known_bug("A3")]
            if env_id in ["Ant-v5", "Humanoid-v5", "HumanoidStandup-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]
            else [],
        )
        for env_id in MUJOCO_ENV_IDS
    ],
)
def test_allow_list_matches_accessor_table(env_id):
    """The parameters the wrapper accepts are exactly those it can get and set.

    ``base.MUJOCO_GETTERS`` builds the allow-list and ``param_look_up`` holds
    the accessors. A name in one but not the other is either rejected although
    it could be applied, or accepted although it cannot.
    """
    cls_name = MUJOCO_ENV_CLASS[env_id]
    assert _settable_params(cls_name) == set(TUNABLE_PARAMS[cls_name])


@pytest.mark.parametrize(
    "env_id",
    [
        pytest.param("Ant-v5", marks=known_bug("A3")),
        pytest.param("Humanoid-v5", marks=known_bug("A3")),
        pytest.param("HumanoidStandup-v5", marks=known_bug("A3")),
        "HalfCheetah-v5",
        "Hopper-v5",
    ],
)
def test_floor_friction_is_tunable(env_id):
    """``floor_friction`` is documented for every env with a floor the robot touches."""
    fn = IncrementUpdate(ContinuousScheduler(), k=0.5)
    ns_env = MujocoWrapper(gym.make(env_id), {"floor_friction": fn})
    ns_env.reset(seed=0)
    before = float(ns_env._get_param_value("floor_friction"))
    ns_env.step(ns_env.action_space.sample())
    assert np.isclose(float(ns_env._get_param_value("floor_friction")), before + 0.5)


# ============================================================
# Parameters are applied on step and restored on reset
# ============================================================

@pytest.mark.parametrize("env_id,param", MUJOCO_ENV_PARAMS, ids=ENV_PARAM_IDS)
def test_mujoco_parameter_update_on_step(env_id, param):
    """One ``step()`` applies exactly one increment to the parameter."""
    cls_name = MUJOCO_ENV_CLASS[env_id]
    env = gym.make(env_id)
    initial_value = TUNABLE_PARAMS[cls_name][param]
    getter, _ = param_look_up(cls_name, param)[0]

    increment_k = 0.234
    update_fn = IncrementUpdate(ContinuousScheduler(start=0), k=increment_k)
    if param == "gravity":
        expected_value = initial_value + np.array([0, 0, increment_k])
    else:
        expected_value = initial_value + increment_k

    ns_env = MujocoWrapper(env, {param: update_fn}, in_sim_change=True)
    ns_env.reset(seed=42)
    ns_env.step(ns_env.action_space.sample())

    new_value = getter(env.unwrapped)
    assert np.allclose(new_value, expected_value), (
        f"Parameter '{param}' in '{env_id}' was not updated correctly. "
        f"Expected: {expected_value}, Got: {new_value}"
    )
    ns_env.close()


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_mujoco_reset_restores_all_parameters(env_id):
    """After stepping, ``reset()`` puts every tunable parameter back to its default."""
    cls_name = MUJOCO_ENV_CLASS[env_id]
    ns_env = _all_params_env(env_id, change_notification=True, delta_change_notification=True)
    obs, _ = ns_env.reset(seed=42)
    assert obs["relative_time"] == 0

    for _ in range(2):
        ns_env.step(ns_env.action_space.sample())

    for param, initial_value in TUNABLE_PARAMS[cls_name].items():
        getter, _ = param_look_up(cls_name, param)[0]
        assert not np.allclose(getter(ns_env.unwrapped), initial_value, atol=1e-6), (
            f"Parameter '{param}' did not change after steps."
        )

    obs, _ = ns_env.reset(seed=42)
    assert obs["relative_time"] == 0, "Relative time did not reset."

    for param, initial_value in TUNABLE_PARAMS[cls_name].items():
        getter, _ = param_look_up(cls_name, param)[0]
        assert np.allclose(getter(ns_env.unwrapped), initial_value, atol=1e-6), (
            f"Parameter '{param}' did not reset to initial value."
        )


@pytest.mark.parametrize(
    "env_id",
    [
        pytest.param(env_id, marks=[known_bug("D1")] if env_id in HUMANOID_IDS else [])
        for env_id in MUJOCO_ENV_IDS
    ],
)
def test_mujoco_reset_observation_is_reproducible(env_id):
    """``reset(seed=s)`` returns the same observation before and after an episode.

    The observation must be computed with the restored parameters. Humanoid
    observations include body masses, so an observation built before the
    restore carries the previous episode's values.
    """
    ns_env = _all_params_env(env_id)
    obs, _ = ns_env.reset(seed=42)
    init_state = obs["state"]

    for _ in range(2):
        ns_env.step(ns_env.action_space.sample())

    obs, _ = ns_env.reset(seed=42)
    assert np.allclose(obs["state"], init_state, atol=1e-6), "State did not reset to initial state."


# ============================================================
# Every parameter changes the simulation
# ============================================================

@pytest.mark.parametrize("env_id,param", MUJOCO_ENV_PARAMS, ids=ENV_PARAM_IDS)
def test_wrapped_vs_unwrapped_dynamics_divergence(env_id, param):
    """A zero increment leaves the dynamics identical to the plain env; a
    non-zero increment makes the trajectory diverge.

    The second half is what keeps a parameter with no physical effect out of
    the allow-list.
    """
    num_steps = 100
    seed = 42

    unwrapped_env = gym.make(env_id)
    env_no_change = MujocoWrapper(
        gym.make(env_id),
        tunable_params={param: IncrementUpdate(ContinuousScheduler(start=0), k=0.0)},
    )
    env_with_change = MujocoWrapper(
        gym.make(env_id),
        tunable_params={param: IncrementUpdate(ContinuousScheduler(start=0), k=0.523)},
    )

    # A fixed action sequence: ``reset(seed=...)`` does not seed the action
    # space, and whether a friction change shows up within 100 steps depends on
    # the trajectory.
    actions = fixed_actions(unwrapped_env, num_steps, seed=seed)

    # --- identical behaviour with a zero increment ---
    obs_unwrapped, _ = unwrapped_env.reset(seed=seed)
    obs_no_change, _ = env_no_change.reset(seed=seed)
    for action in actions:
        obs_unwrapped, _, done_unwrapped, _, _ = unwrapped_env.step(action)
        obs_no_change, _, done_no_change, _, _ = env_no_change.step(action)
        if done_unwrapped or done_no_change:
            break

    assert np.allclose(obs_no_change["state"], obs_unwrapped, atol=1e-6), (
        f"[{env_id}/{param}] Final states diverged after {num_steps} steps with k=0. "
        "Expected identical dynamics."
    )

    # --- divergent behaviour with a non-zero increment ---
    unwrapped_env.reset(seed=seed)
    obs_with_change, _ = env_with_change.reset(seed=seed)
    for action in actions:
        obs_unwrapped, _, done_unwrapped, _, _ = unwrapped_env.step(action)
        obs_with_change, _, done_with_change, _, _ = env_with_change.step(action)
        if done_unwrapped or done_with_change:
            break

    assert not np.allclose(obs_with_change["state"], obs_unwrapped, atol=1e-6), (
        f"[{env_id}/{param}] Final states did NOT diverge after {num_steps} steps with k=0.523. "
        "Wrapper failed to change dynamics."
    )

    unwrapped_env.close()
    env_no_change.close()
    env_with_change.close()


@pytest.mark.parametrize(
    "env_id",
    [
        pytest.param(env_id, marks=[known_bug("D4")] if env_id in ["Ant-v5", "Humanoid-v5"] else [])
        for env_id in MUJOCO_ENV_IDS
    ],
)
def test_unchanged_parameters_leave_rewards_identical_to_gymnasium(env_id):
    """Wrapping an env without changing any parameter must not change its rewards.

    States already match (see the test above). Rewards are checked separately
    because Ant and Humanoid derive a velocity reward from body positions that
    the wrapper recomputes before every step.
    """
    param = next(p for p in TUNABLE_PARAMS[MUJOCO_ENV_CLASS[env_id]] if p != "gravity")
    plain = gym.make(env_id)
    wrapped = MujocoWrapper(gym.make(env_id), {param: NoUpdate(ContinuousScheduler())})
    plain.reset(seed=0)
    wrapped.reset(seed=0)

    rng = np.random.default_rng(0)
    for _ in range(30):
        action = rng.uniform(plain.action_space.low, plain.action_space.high).astype(np.float32)
        _, reward_plain, term_plain, _, _ = plain.step(action)
        _, reward_wrapped, term_wrapped, _, _ = wrapped.step(action)
        assert np.isclose(reward_plain, reward_wrapped), "reward differs from the unwrapped env"
        if term_plain or term_wrapped:
            break


# ============================================================
# Integration state survives a parameter update (mj_setConst)
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
    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    params = list(TUNABLE_PARAMS[env_name].keys())
    mass_param = next(p for p in params if "mass" in p)

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
