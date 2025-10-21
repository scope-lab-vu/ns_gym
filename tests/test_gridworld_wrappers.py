import pytest
import gymnasium as gym
import numpy as np
import copy
import ns_gym
from ns_gym.base import UpdateFn, TUNABLE_PARAMS
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import DistributionIncrementUpdate
from ns_gym.wrappers import NSCliffWalkingWrapper, NSFrozenLakeWrapper #, NSBridgeWrapper # Import your wrappers

SUPPORTED_GRID_WORLD_ENV_IDS = [
    "CliffWalking-v1",
    "FrozenLake-v1",
    # "ns_gym/Bridge-v0" # Uncomment when Bridge wrapper is ready
]

@pytest.fixture
def gridworld_wrappers():
    """Maps environment IDs to their corresponding wrapper classes."""
    return {
        "CliffWalking-v1": NSCliffWalkingWrapper,
        "FrozenLake-v1": NSFrozenLakeWrapper,
        # "ns_gym/Bridge-v0": NSBridgeWrapper,
    }

@pytest.fixture
def gridworld_params():
    """Returns a dictionary of valid tunable parameters for each gridworld environment."""
    # For gridworlds, the non-stationarity is often in the transition probabilities 'P'
    update_fn = DistributionIncrementUpdate(ContinuousScheduler(), k=-0.1)
    return {
        "CliffWalking-v1": {"P": update_fn},
        "FrozenLake-v1": {"P": update_fn},
        # "ns_gym/Bridge-v0": {"P": update_fn}, # Or other custom params
    }

@pytest.mark.parametrize("env_id", SUPPORTED_GRID_WORLD_ENV_IDS)
def test_gridworld_wrapper_init(gridworld_wrappers, gridworld_params, env_id):
    """Tests the basic initialization of the gridworld wrappers."""
    env = gym.make(env_id)
    tunable_params = gridworld_params[env_id]
    WrapperClass = gridworld_wrappers[env_id]
    
    ns_env = WrapperClass(env, tunable_params, change_notification=True, delta_change_notification=True)
    
    assert isinstance(ns_env, WrapperClass)
    assert isinstance(ns_env.unwrapped, gym.Env)
    assert ns_env.change_notification
    assert ns_env.delta_change_notification
    assert len(ns_env.tunable_params) == len(tunable_params)
    assert all(isinstance(fn, UpdateFn) for fn in ns_env.tunable_params.values())

@pytest.mark.parametrize("env_id", SUPPORTED_GRID_WORLD_ENV_IDS)
def test_observation_structure(gridworld_wrappers, gridworld_params, env_id):
    """Tests the observation structure after reset and step for a sample gridworld."""
    env = gym.make(env_id)

    wrapper = gridworld_wrappers[env_id]
    tunable_params = gridworld_params[env_id]

    update_fn = DistributionIncrementUpdate(ContinuousScheduler(), k=1.0)
    tunable_params = {"P": update_fn}

    ns_env = wrapper(env, tunable_params, change_notification=True, delta_change_notification=True,)
    obs, info = ns_env.reset(seed=42)
    
    assert isinstance(obs, dict)
    assert all(k in obs for k in ['state', 'env_change', 'delta_change', 'relative_time'])
    assert obs['relative_time'] == 0
    assert not obs['env_change']['P']
    assert obs['delta_change']['P'] == 0.0
    
    action = ns_env.action_space.sample()
    obs, _, _, _, _ = ns_env.step(action)
    
    assert obs['relative_time'] > 0
    assert isinstance(obs['state'], int)

@pytest.mark.parametrize("env_id", SUPPORTED_GRID_WORLD_ENV_IDS)
def test_deepcopy_divergence(gridworld_wrappers, gridworld_params, env_id):
    """Tests that a deepcopied sim_env has a static transition model while the original evolves."""
    env = gym.make(env_id)
    tunable_params = gridworld_params[env_id]
    WrapperClass = gridworld_wrappers[env_id]
    
    ns_env = WrapperClass(env, tunable_params, change_notification=True, delta_change_notification=True)
    ns_env.reset(seed=42)
    
    action = ns_env.action_space.sample()

    obs_original, _, _, _, _ = ns_env.step(action)
    
    sim_env = copy.deepcopy(ns_env)

    assert isinstance(sim_env, WrapperClass)
    assert sim_env.is_sim_env
    assert sim_env.unwrapped.P.keys() == ns_env.unwrapped.P.keys()

@pytest.mark.parametrize("env_id", SUPPORTED_GRID_WORLD_ENV_IDS)
def test_get_planning_env(gridworld_wrappers, gridworld_params, env_id):
    """Tests that get_planning_env returns a deep copy with is_sim_env=True."""
    env = gym.make(env_id)
    tunable_params = gridworld_params[env_id]
    WrapperClass = gridworld_wrappers[env_id]
    
    ns_env = WrapperClass(env, tunable_params, change_notification=True, delta_change_notification=True)
    ns_env.reset(seed=42)

    done = True
    while done:
        obs,rew,done,truncated,info = ns_env.step(ns_env.action_space.sample())  # Take a step to change the environment
        if done:
            ns_env.reset()
    
    planning_env = ns_env.get_planning_env()
    
    assert isinstance(planning_env, WrapperClass)
    assert planning_env.is_sim_env
    assert planning_env.unwrapped.P == ns_env.unwrapped.P
    assert planning_env.tunable_params.keys() == ns_env.tunable_params.keys()

    env = gym.make(env_id)
    tunable_params = gridworld_params[env_id]
    WrapperClass = gridworld_wrappers[env_id]
    ns_env = WrapperClass(env, tunable_params, change_notification=False, delta_change_notification=False)
    ns_env.reset(seed=42)

    done = True
    while done:
        obs,rew,done,truncated,info = ns_env.step(ns_env.action_space.sample())  # Take a step to change the environment
        if done:
            ns_env.reset()

    planning_env = ns_env.get_planning_env()
    assert isinstance(planning_env, WrapperClass)
    assert planning_env.is_sim_env
    assert planning_env.unwrapped.P != ns_env.unwrapped.P



@pytest.mark.parametrize("env_id", SUPPORTED_GRID_WORLD_ENV_IDS)    
def test_invalid_tunable_param_gridworld(env_id, gridworld_wrappers):
    """Tests that the wrapper raises an error for an invalid parameter name."""
    env = gym.make(env_id)
    update_fn = DistributionIncrementUpdate(ContinuousScheduler(), k=0.1)
    tunable_params = {"invalid_param": update_fn} # Invalid parameter

    with pytest.raises(AssertionError):
        ns_env = gridworld_wrappers[env_id](env, tunable_params)