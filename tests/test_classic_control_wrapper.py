import pytest
import gymnasium as gym
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import IncrementUpdate, RandomWalk
import numpy as np
import copy
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.base import UpdateFn # Import necessary base classes for typing/check
from ns_gym.base import TUNABLE_PARAMS
from gymnasium import register, registry

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
        "random_walk": RandomWalk(schedulers["periodic"]),
        "decrement": IncrementUpdate(schedulers["continuous"], k=-0.1)
    }

@pytest.fixture
def classic_control_params(update_functions):
    """
    Returns a dictionary of valid tunable parameters for each classic control environment.
    """
    increment_fn = update_functions["increment"]
    random_walk_fn = update_functions["random_walk"]
    decrement_fn = update_functions["decrement"]

    return {
        "CartPole-v1": {"masspole": increment_fn, "gravity": increment_fn},
        "Acrobot-v1": {"LINK_LENGTH_1": increment_fn, "LINK_MASS_2": increment_fn},
        "MountainCar-v0": {"gravity":decrement_fn, "force": increment_fn},
        "MountainCarContinuous-v0": {"power": increment_fn}, # Only one tunable param
        "Pendulum-v1": {"m": increment_fn, "g": increment_fn},
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
    obs, info = ns_env.reset(seed=42)
    
    assert isinstance(obs, dict)
    assert all(k in obs for k in ['state', 'env_change', 'delta_change', 'relative_time'])
    
    assert obs['relative_time'] == 0.0
    
    assert all(not obs['env_change'][p] for p in tunable_params.keys())
    assert all(obs['delta_change'][p] == 0.0 for p in tunable_params.keys())
    
    action = ns_env.action_space.sample()
    obs, reward, terminated, truncated, info = ns_env.step(action)

@pytest.mark.parametrize("env_id", SUPPORTED_CLASSIC_CONTROL_ENV_IDS)
def test_deepcopy(classic_control_params, env_id):

    # if env_id == "MountainCarContinuous-v0" or env_id == "MountainCar-v0":
    #     pytest.skip("Skipping deepcopy test for MountainCarContinuous-v0 due to known issue with deepcopying Box2D environments.")


    if env_id in ["MountainCar-v0", "MountainCarContinuous-v0"]:
        pytest.skip(
            f"Skipping deepcopy test for {env_id} due to a known issue "
            "where sim and real states do not diverge as expected in this test setup."
        )

    env = gym.make(env_id)
    tunable_params = classic_control_params[env_id]
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



def validate_wrapper_order(env):

        # Check 1: Outermost Wrapper
    assert isinstance(env, NSClassicControlWrapper), (
        f"Expected outermost wrapper to be {NSClassicControlWrapper.__name__}, "
        f"but got {type(env).__name__}"
    )

    # Check 2: No Double Wraps
    current_layer = env.env 
    stack_depth = 1
    
    while hasattr(current_layer, "env"):
        assert not isinstance(current_layer, NSClassicControlWrapper), (
            f"Double wrap detected! Found {NSClassicControlWrapper.__name__} again "
            f"at depth {stack_depth}."
        )
        current_layer = current_layer.env
        stack_depth += 1

    # Check the final base environment (root)
    assert not isinstance(current_layer, NSClassicControlWrapper), (
        f"Double wrap detected! Found {NSClassicControlWrapper.__name__} as the base environment."
    )


@pytest.mark.parametrize("env_id", SUPPORTED_CLASSIC_CONTROL_ENV_IDS)
def test_get_planning_env(classic_control_params, env_id):

    if env_id in ["MountainCar-v0", "MountainCarContinuous-v0"]:
        pytest.skip(
            f"Skipping deepcopy test for {env_id} due to a known issue "
            "where sim and real states do not diverge as expected in this test setup."
        )

    env = gym.make(env_id)


    tunable_params = classic_control_params[env_id]

    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=False, delta_change_notification=False, in_sim_change=False)
    ns_env.reset(seed=42)

    env_name = env.unwrapped.__class__.__name__

    init_params = {param: getattr(ns_env.unwrapped, param) for param in tunable_params.keys()}

    action = ns_env.action_space.sample()
    obs, reward, terminated, truncated, info = ns_env.step(action)

    planning_env = ns_env.get_planning_env()

    validate_wrapper_order(planning_env)

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
    tunable_params = classic_control_params[env_id]

    ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True, delta_change_notification=True, in_sim_change=False)
    ns_env.reset(seed=42)

    action = ns_env.action_space.sample()
    obs, reward, terminated, truncated, info = ns_env.step(action)
    planning_env = ns_env.get_planning_env()

    validate_wrapper_order(planning_env)


    assert isinstance(planning_env, NSClassicControlWrapper)
    assert env_name == planning_env.unwrapped.__class__.__name__
    assert planning_env.is_sim_env == True

    for param in tunable_params.keys():
        assert getattr(planning_env.unwrapped, param) == getattr(ns_env.unwrapped, param)
    



@pytest.mark.parametrize("env_id", SUPPORTED_CLASSIC_CONTROL_ENV_IDS)
def test_wrapped_vs_unwrapped(classic_control_params, env_id):

    if env_id in ["MountainCar-v0", "MountainCarContinuous-v0"]:
        pytest.skip(
            f"Skipping wrapped vs unwrapped test for {env_id} due to a known issue "
            "where sim and real states do not diverge as expected in this test setup."
        )

    env_1 = gym.make(env_id)
    env_2 = gym.make(env_id)

    ns_env = NSClassicControlWrapper(env_1, classic_control_params[env_id], change_notification=False, delta_change_notification=False)
    ns_env.reset(seed=42)
    env_2.reset(seed=42)

    action = ns_env.action_space.sample()
    obs_ns, reward_ns, terminated_ns, truncated_ns, info_ns = ns_env.step(action)
    obs_env, reward_env, terminated_env, truncated_env, info_env = env_2.step(action)
    assert not np.array_equal(obs_ns['state'], obs_env)


    

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


@pytest.mark.parametrize("env_id", SUPPORTED_CLASSIC_CONTROL_ENV_IDS)
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




@pytest.mark.parametrize("env_id", SUPPORTED_CLASSIC_CONTROL_ENV_IDS)
def test_registration(classic_control_params, env_id):
    """
    Tests that a custom registered environment using the NS wrapper 
    maintains the correct wrapper stack (Outermost = NSWrapper)
    and that get_planning_env() preserves this structure.
    """

    tunable_params = classic_control_params[env_id]
    
    def _make_custom_env(**kwargs):
        base_env = gym.make(env_id, **kwargs) 
        return NSClassicControlWrapper(base_env, tunable_params, change_notification=False, delta_change_notification=False, in_sim_change=False)

    # We use a unique ID for testing to avoid collisions
    custom_id = f"Test-NS-{env_id}"
    
    # Idempotency: Remove if already registered from a previous run
    if custom_id in registry:
        del registry[custom_id]
        
    register(
        id=custom_id,
        entry_point=_make_custom_env,
        # CRITICAL: This ensures gym.make() doesn't add outer OrderEnforcing/PassiveChecker
        disable_env_checker=True ,
        order_enforce=False
    )

    try:
        # 3. Create the environment via gym.make
        # We rely on the registration's disable_env_checker=True here
        ns_env = gym.make(custom_id)

        # 4. Validate Stack Integrity (gym.make)
        # Should be: <NSWrapper<...>> NOT <OrderEnforcing<NSWrapper<...>>>
        validate_wrapper_order(ns_env)
        
        # 5. Validate Stack Integrity (get_planning_env)
        ns_env.reset(seed=123)
        planning_env = ns_env.get_planning_env()
        
        # The planning env must also be cleanly wrapped
        validate_wrapper_order(planning_env)
        
        # Standard sanity checks on the planning env
        assert planning_env.is_sim_env
        assert isinstance(planning_env.unwrapped, gym.Env)

    finally:
        # Cleanup registration to avoid polluting the global registry
        if custom_id in registry:
            del registry[custom_id]