"""Reproducibility: what ``reset(seed=...)`` controls and what stays random."""


import gymnasium as gym
import numpy as np
import pytest

from ns_gym.base import TUNABLE_PARAMS
from ns_gym.schedulers import (
    ContinuousScheduler,
    DecayingProbabilityScheduler,
    MemorylessScheduler,
    RandomScheduler,
)
from ns_gym.update_functions import LCBoundedDistrubutionUpdate, RandomCategorical, RandomWalk
from ns_gym.wrappers import MujocoWrapper, NSClassicControlWrapper
from ns_gym.wrappers.mujoco_env import param_look_up

from _helpers import (
    BRIDGE_ENV_ID,
    CASE_BY_ID,
    CC_SEEDING_PARAMS,
    CLASSIC_CONTROL_ENV_IDS,
    DISTRIBUTION_CASES,
    GRIDWORLD_ENV_IDS,
    MUJOCO_TIME_TEST_IDS,
    env_cases,
    fixed_actions,
    known_bug,
    safe_actions,
    same,
)


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


@pytest.mark.parametrize("env_id", MUJOCO_TIME_TEST_IDS)
def test_reset_no_seed_different_rng_sequence_mujoco(env_id):
    """MuJoCo: reset() with no seed should produce different RNG sequences."""

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
# Random schedulers: when a change happens is random too
# ============================================================

STOCHASTIC_SCHEDULERS = {
    "RandomScheduler": lambda: RandomScheduler(probability=0.3),
    "MemorylessScheduler": lambda: MemorylessScheduler(p=0.2),
    "DecayingProbabilityScheduler": lambda: DecayingProbabilityScheduler(
        initial_probability=0.5, decay_rate=0.001
    ),
}


def _firing_pattern(env, n=150):
    """Which of the next ``n`` steps changed the parameter (Pendulum never terminates)."""
    pattern = []
    for action in fixed_actions(env, n):
        _, _, _, _, info = env.step(action)
        pattern.append(info["Ground Truth Env Change"]["m"])
    return pattern


def _pendulum_with(scheduler):
    return CASE_BY_ID["Pendulum-v1"].make(scheduler=scheduler)


@pytest.mark.parametrize("name", STOCHASTIC_SCHEDULERS)
def test_random_schedule_same_seed_is_reproducible(name):
    env = _pendulum_with(STOCHASTIC_SCHEDULERS[name]())
    env.reset(seed=7)
    first = _firing_pattern(env)
    env.reset(seed=7)
    assert _firing_pattern(env) == first
    assert 0 < sum(first) < len(first), "the scheduler should fire on some steps but not all"


@pytest.mark.parametrize("name", STOCHASTIC_SCHEDULERS)
def test_random_schedule_depends_on_seed(name):
    """``reset(seed=...)`` must reach the scheduler's RNG, not only the update function's."""
    env = _pendulum_with(STOCHASTIC_SCHEDULERS[name]())
    env.reset(seed=1)
    first = _firing_pattern(env)
    env.reset(seed=2)
    assert _firing_pattern(env) != first, "seeds 1 and 2 produced the same change schedule"


@pytest.mark.parametrize("name", STOCHASTIC_SCHEDULERS)
def test_random_schedule_differs_between_unseeded_episodes(name):
    """Without a seed each episode draws a new schedule, as the update functions do."""
    env = _pendulum_with(STOCHASTIC_SCHEDULERS[name]())
    env.reset(seed=0)
    first = _firing_pattern(env)
    env.reset()
    second = _firing_pattern(env)
    env.reset()
    third = _firing_pattern(env)
    assert first != second and second != third, "the same change schedule was replayed"


# ============================================================
# Transition-probability envs with a random update, Bridge included
# ============================================================

def _first_random_dist(case, env, **reset_kwargs):
    """The distribution in force after the first step of a fresh episode."""
    env.reset(**reset_kwargs)
    env.step(safe_actions(case, env, 1)[0])
    return case.executed(env)


def _random_categorical_env(case):
    return case.make(RandomCategorical(ContinuousScheduler()))


@pytest.mark.parametrize("case", env_cases(DISTRIBUTION_CASES))
def test_random_distribution_same_seed_is_reproducible(case):
    env = _random_categorical_env(case)
    assert same(_first_random_dist(case, env, seed=3), _first_random_dist(case, env, seed=3))


@pytest.mark.parametrize("case", env_cases(DISTRIBUTION_CASES, known_bugs={BRIDGE_ENV_ID: "B3"}))
def test_random_distribution_depends_on_seed(case):
    env = _random_categorical_env(case)
    assert not same(_first_random_dist(case, env, seed=1), _first_random_dist(case, env, seed=2))


@pytest.mark.parametrize("case", env_cases(DISTRIBUTION_CASES, known_bugs={BRIDGE_ENV_ID: "B3"}))
def test_random_distribution_differs_between_unseeded_episodes(case):
    env = _random_categorical_env(case)
    first = _first_random_dist(case, env, seed=0)
    second = _first_random_dist(case, env)
    third = _first_random_dist(case, env)
    assert not same(first, second) and not same(second, third)


def test_lc_bounded_update_is_seeded_by_reset():
    """The Lipschitz-bounded update draws from an inner update function's RNG.

    ``L=2`` accepts every candidate (2 is the largest possible distance between
    two distributions over three outcomes), so this runs quickly.
    """
    case = CASE_BY_ID["FrozenLake-v1"]
    env = case.make(LCBoundedDistrubutionUpdate(ContinuousScheduler(), L=2.0))
    assert not same(_first_random_dist(case, env, seed=1), _first_random_dist(case, env, seed=2))


# ============================================================
# The Bridge base env
# ============================================================

def _bridge_states(seed, n=40):
    env = gym.make(BRIDGE_ENV_ID, global_init_probs=[0.4, 0.3, 0.3])
    env.reset(seed=seed)
    states = []
    for _ in range(n):
        state, _, terminated, truncated, _ = env.step(2)
        states.append(state)
        if terminated or truncated:
            env.reset()
    return states


@known_bug("B2")
def test_bridge_env_same_seed_is_reproducible():
    """Slips must be drawn from the env's seeded RNG, like every gymnasium env."""
    assert _bridge_states(seed=0) == _bridge_states(seed=0)
