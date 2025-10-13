import ns_gym
import pytest
import gymnasium as gym
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import IncrementUpdate, RandomWalk
import numpy as np
import copy
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.base import NSWrapper, UpdateFn # Import necessary base classes for typing/check



SUPPORTED_CLASSIC_CONTROL_ENV_IDS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
]


@pytest.fixture
def schedulers():
    return {
        "continuous": ContinuousScheduler(),
        "periodic": PeriodicScheduler(period=3)
    }

@pytest.fixture
def update_functions(schedulers):
    return {
        "increment": IncrementUpdate(schedulers["continuous"], k=0.1),
        "random_walk": RandomWalk(schedulers["periodic"])
    }

@pytest.fixture
def classic_control_params(update_functions):
    """
    Returns a dictionary of valid tunable parameters for each classic control environment.
    """
    increment_fn = update_functions["increment"]
    random_walk_fn = update_functions["random_walk"]

    return {
        "CartPole-v1": {"masspole": increment_fn, "gravity": random_walk_fn},
        "Acrobot-v1": {"LINK_LENGTH_1": increment_fn, "LINK_MASS_2": random_walk_fn},
        "MountainCar-v0": {"gravity": random_walk_fn, "force": increment_fn},
        "MountainCarContinuous-v0": {"power": increment_fn}, # Only one tunable param
        "Pendulum-v1": {"m": increment_fn, "g": random_walk_fn},
    }

# --- TEST NSClassicControlWrapper ---

@pytest.mark.parametrize("env_id", SUPPORTED_CLASSIC_CONTROL_ENV_IDS)
def test_classic_control_wrapper(classic_control_params, env_id):
    """Tests the basic initialization of the wrapper for classic control environments."""

    env = gym.make(env_id)
    tunable_params = classic_control_params[env_id]

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

def test_step_and_observation_structure(classic_control_params):
    """Tests the observation structure and change logic (notification ON/OFF)."""
    

    env_id = "CartPole-v1"
    env = gym.make(env_id)
    tunable_params = classic_control_params[env_id]
    

    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True, delta_change_notification=True)
    obs, info = ns_env.reset()
    
    assert isinstance(obs, dict)
    assert all(k in obs for k in ['state', 'env_change', 'delta_change', 'relative_time'])
    
    assert obs['relative_time'] == 0.0
    
    assert all(not obs['env_change'][p] for p in tunable_params.keys())
    assert all(obs['delta_change'][p] == 0.0 for p in tunable_params.keys())
    
    action = ns_env.action_space.sample()
    obs, reward, terminated, truncated, info = ns_env.step(action)


def test_deepcopy(classic_control_params):
    env = gym.make("CartPole-v1")
    tunable_params = classic_control_params["CartPole-v1"]
    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True, delta_change_notification=True)
    ns_env.reset()
    
    action = ns_env.action_space.sample()
    for t in range(5):
        ns_env.step(action)

    sim_env = copy.deepcopy(ns_env)

    assert isinstance(sim_env, NSClassicControlWrapper)
    assert sim_env.is_sim_env == True

    from ns_gym.base import TUNABLE_PARAMS


    


def test_get_planning_env(classic_control_params):
    env = gym.make("CartPole-v1")
    tunable_params = classic_control_params["CartPole-v1"]
    
    ns_env_notify = NSClassicControlWrapper(env, tunable_params, change_notification=True)
    ns_env_notify.reset()

    planning_env = ns_env_notify.get_planning_env()
    assert isinstance(planning_env, NSClassicControlWrapper)

    assert planning_env.is_sim_env == True
    

def test_dependency_resolver(classic_control_params):
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



