"""Behaviour specific to ``NSClassicControlWrapper``: parameters, constraints, derived quantities."""

import copy

import gymnasium as gym
import numpy as np
import pytest

from ns_gym.base import TUNABLE_PARAMS, UpdateFn
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate
from ns_gym.wrappers import NSClassicControlWrapper

from _helpers import (
    CASE_BY_ID,
    CLASSIC_CONTROL_ENV_IDS,
    safe_actions,
    validate_no_double_wrap,
)


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_classic_control_wrapper(cc_params, env_id):
    """Tests the basic initialization of the wrapper for classic control environments."""

    env = gym.make(env_id)
    tunable_params = cc_params[env_id]

    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True)
    env_change_space = gym.spaces.Dict({param_name: gym.spaces.Discrete(2) for param_name in tunable_params.keys()}) 
    delta_change_space = gym.spaces.Dict({param_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=()) for param_name in tunable_params.keys()})


    assert isinstance(ns_env, NSClassicControlWrapper)
    assert isinstance(ns_env.unwrapped, gym.Env)
    assert ns_env.change_notification == True
    assert ns_env.delta_change_notification == False
    # Check that the number of tunable parameters matches the input
    assert len(ns_env.tunable_params) == len(tunable_params)
    # Check that the update functions are instances of UpdateFn subclasses
    assert all(isinstance(fn, UpdateFn) for fn in ns_env.tunable_params.values())
    assert ns_env.is_sim_env == False
    assert ns_env.observation_space == gym.spaces.Dict({
        "state": ns_env.unwrapped.observation_space,
        "env_change": env_change_space,
        "delta_change": delta_change_space,
        "relative_time": gym.spaces.Box(low=0, high=np.inf, shape=())
    })


# --- TEST STEP AND OBSERVATION STRUCTURE ---


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy(cc_params, env_id):

    # if env_id == "MountainCarContinuous-v0" or env_id == "MountainCar-v0":
    #     pytest.skip("Skipping deepcopy test for MountainCarContinuous-v0 due to known issue with deepcopying Box2D environments.")


    env = gym.make(env_id)
    tunable_params = cc_params[env_id]
    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True, delta_change_notification=True)
    ns_env.reset(seed=42)
    
    action = ns_env.action_space.sample()
    for t in range(5):
        ns_env.step(action)

    sim_env = copy.deepcopy(ns_env)

    assert isinstance(sim_env, NSClassicControlWrapper)
    assert sim_env.is_sim_env == True

    env_name = env.unwrapped.__class__.__name__

    assert sim_env.unwrapped.__class__.__name__ == env_name

    for param in TUNABLE_PARAMS[env_name]:
        assert getattr(sim_env.unwrapped, param) == getattr(ns_env.unwrapped, param)

    env_action = sim_env.action_space.sample()


    sim_obs, sim_reward, sim_terminated, sim_truncated, sim_info = sim_env.step(env_action)
    obs, reward, terminated, truncated, info = ns_env.step(env_action)


    assert not np.array_equal(sim_obs['state'], obs['state'])  # States should differ after a step


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_get_planning_env(cc_params, env_id):


    env = gym.make(env_id)


    tunable_params = cc_params[env_id]

    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=False, delta_change_notification=False, in_sim_change=False)
    ns_env.reset(seed=42)

    env_name = env.unwrapped.__class__.__name__

    init_params = {param: getattr(ns_env.unwrapped, param) for param in tunable_params.keys()}

    action = ns_env.action_space.sample()
    obs, reward, terminated, truncated, info = ns_env.step(action)

    planning_env = ns_env.get_planning_env()

    validate_no_double_wrap(planning_env, NSClassicControlWrapper)

    assert isinstance(planning_env, NSClassicControlWrapper)
    assert env_name == planning_env.unwrapped.__class__.__name__
    assert planning_env.is_sim_env == True

    for param in tunable_params.keys():
        assert getattr(planning_env.unwrapped, param) != getattr(ns_env.unwrapped, param)

    action = planning_env.action_space.sample()

    planning_obs, planning_reward, planning_terminated, planning_truncated, planning_info = planning_env.step(action)
    obs, reward, terminated, truncated, info = ns_env.step(action)

    assert not np.array_equal(planning_obs['state'], obs['state'])  # States should differ after a step]

    env = gym.make(env_id)
    tunable_params = cc_params[env_id]

    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True, delta_change_notification=True, in_sim_change=False)
    ns_env.reset(seed=42)

    action = ns_env.action_space.sample()
    obs, reward, terminated, truncated, info = ns_env.step(action)
    planning_env = ns_env.get_planning_env()

    validate_no_double_wrap(planning_env, NSClassicControlWrapper)


    assert isinstance(planning_env, NSClassicControlWrapper)
    assert env_name == planning_env.unwrapped.__class__.__name__
    assert planning_env.is_sim_env == True

    for param in tunable_params.keys():
        assert getattr(planning_env.unwrapped, param) == getattr(ns_env.unwrapped, param)
    


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_wrapped_vs_unwrapped(cc_params, env_id):


    env_1 = gym.make(env_id)
    env_2 = gym.make(env_id)

    ns_env = NSClassicControlWrapper(env_1, cc_params[env_id], change_notification=False, delta_change_notification=False)
    ns_env.reset(seed=42)
    env_2.reset(seed=42)

    action = ns_env.action_space.sample()
    obs_ns, reward_ns, terminated_ns, truncated_ns, info_ns = ns_env.step(action)
    obs_env, reward_env, terminated_env, truncated_env, info_env = env_2.step(action)
    assert not np.array_equal(obs_ns['state'], obs_env)


    

def test_dependency_resolver(cc_params):
    env = gym.make("CartPole-v1")
    
    scheduler = ContinuousScheduler(start=0)

    update_fn_masspole = IncrementUpdate(scheduler, k=0.5) 
    
    tunable_params = {"masspole": update_fn_masspole}
    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=False)
    ns_env.reset()
    
    initial_masspole = env.unwrapped.masspole # 0.1
    initial_masscart = env.unwrapped.masscart # 1.0
    initial_length = env.unwrapped.length # 0.5
    
    initial_total_mass = initial_masspole + initial_masscart # 1.1
    initial_polemass_length = initial_masspole * initial_length # 0.05
    
    ns_env.step(0)
    
    assert env.unwrapped.total_mass != initial_total_mass
    assert np.isclose(env.unwrapped.total_mass, 1.6)
    assert np.isclose(env.unwrapped.polemass_length, 0.3)


def test_invalid_tunable_param():

    env = gym.make("CartPole-v1")
    scheduler = ContinuousScheduler(start=0)
    update_fn = IncrementUpdate(scheduler, k=0.5) 
    tunable_params = {"invalid_param": update_fn}  # Invalid parameter name

    with pytest.raises(AssertionError, match="Tunable parameters .* not all in default tunable parameters .* for environment .*"):
        ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=False)


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_valid_tunable_param(env_id):
    

    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    scheduler = ContinuousScheduler(start=0)
    update_fn = IncrementUpdate(scheduler, k=0.5) 
    valid_params = TUNABLE_PARAMS[env_name]    
    
    for param in valid_params.keys():
        tunable_params = {param: update_fn}  # Valid parameter name
        ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=False)
        assert isinstance(ns_env, NSClassicControlWrapper)


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
# Derived quantities, constraints and copy behaviour
# ============================================================

def test_cartpole_reset_restores_derived_quantities():
    """``reset()`` restores ``masspole``; ``total_mass`` and ``polemass_length`` must follow."""
    case = CASE_BY_ID["CartPole-v1"]
    env = case.make()
    env.reset(seed=0)
    for action in safe_actions(case, env, 4):
        env.step(action)

    env.reset(seed=0)
    u = env.unwrapped
    assert np.isclose(u.total_mass, u.masspole + u.masscart)
    assert np.isclose(u.polemass_length, u.masspole * u.length)


@pytest.mark.parametrize("link", ["1", "2"])
def test_acrobot_link_cannot_shrink_below_its_centre_of_mass(link):
    """A link shorter than the distance to its own centre of mass is unphysical."""
    length, com = f"LINK_LENGTH_{link}", f"LINK_COM_POS_{link}"
    fn = IncrementUpdate(ContinuousScheduler(), k=-0.3)
    env = NSClassicControlWrapper(gym.make("Acrobot-v1"), {length: fn})
    env.reset(seed=0)
    for _ in range(3):
        env.step(0)

    assert getattr(env.unwrapped, length) >= getattr(env.unwrapped, com)


@pytest.mark.parametrize("env_id", CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy_does_not_warn(cc_params, env_id):
    """Planners copy the env thousands of times; a copy must not emit a warning."""
    import warnings

    env = NSClassicControlWrapper(gym.make(env_id), cc_params[env_id])
    env.reset(seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        copy.deepcopy(env)

    assert not caught, f"deepcopy warned: {caught[0].message}"
