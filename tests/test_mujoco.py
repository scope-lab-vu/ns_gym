import pytest
import gymnasium as gym
import ns_gym
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate
import numpy as np



SUPPORTED_MUJOCO_ENV_IDS = [
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


@pytest.mark.parametrize("env_id", SUPPORTED_MUJOCO_ENV_IDS)
def test_mujoco_wrapper_init(env_id):
    # Import MujocoWrapper from its correct module
    from ns_gym.wrappers.mujoco_env import MujocoWrapper
    assert MujocoWrapper is not None


    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__
    
    tunable_params = list(ns_gym.base.TUNABLE_PARAMS[env_name].keys())

    param_map = {param: ns_gym.update_functions.IncrementUpdate(ns_gym.schedulers.ContinuousScheduler(), k=0.1) for param in tunable_params}

    ns_env = MujocoWrapper(env, param_map, change_notification=True, delta_change_notification=True)
    assert isinstance(ns_env, MujocoWrapper)



    
@pytest.mark.parametrize("env_id", SUPPORTED_MUJOCO_ENV_IDS)
def test_attribute_accessibility(env_id):
    """s
    For a given env_id, checks if all mapped tunable parameters are
    actually accessible on the environment object.
    """
    print(f"\n--- Testing Environment: {env_id} ---")


    from ns_gym.base import TUNABLE_PARAMS
    from ns_gym.wrappers.mujoco_env import MujocoWrapper, param_look_up
    assert MujocoWrapper is not None
    assert param_look_up is not None

    assert TUNABLE_PARAMS is not None


    env = gym.make(env_id)
    env_name = env.unwrapped.__class__.__name__

    tunable_params = list(ns_gym.base.TUNABLE_PARAMS[env_name].keys())

    for param in tunable_params:
        fn_list = param_look_up(env_name, param)

        for fn in fn_list:
            getter, setter = fn
            assert callable(getter), f"Getter for {param} in {env_name} is not callable"
            assert callable(setter), f"Setter for {param} in {env_name} is not callable"
    
@pytest.mark.parametrize("env_id", SUPPORTED_MUJOCO_ENV_IDS)
def test_mujoco_parameter_update_on_step(env_id):
    """
    Tests that each tunable parameter for a given MuJoCo environment is correctly
    updated after a single `step()` call.
    """

    from ns_gym.wrappers.mujoco_env import MujocoWrapper, param_look_up
    from ns_gym.base import TUNABLE_PARAMS
    assert MujocoWrapper is not None
    assert TUNABLE_PARAMS is not None
    # Get the list of friendly names for all tunable parameters for this environment
    env_name = gym.make(env_id).unwrapped.__class__.__name__
    all_param_names = list(TUNABLE_PARAMS[env_name].keys())

    print(f"\n--- Testing updates for {env_id} ({len(all_param_names)} params) ---")

    for param_name in all_param_names:
        env = gym.make(env_id)

        initial_value = TUNABLE_PARAMS[env_name][param_name]
        print(f"  Testing parameter: '{param_name}/{env_name}' (initial value: {initial_value})")
        # initial_value = getter(env.unwrapped
        getter, setter = param_look_up(env_name, param_name)[0]

        increment_k = 0.234  
        scheduler = ContinuousScheduler(start=0)
        update_fn = IncrementUpdate(scheduler, k=increment_k)

        if param_name == "gravity":
            expected_value = initial_value + np.array([0, 0, increment_k])
        else:
            expected_value = initial_value + increment_k

        tunable_params = {param_name: update_fn}
        ns_env = MujocoWrapper(env, tunable_params, in_sim_change=True)
        ns_env.reset(seed=42)

        ns_env.step(ns_env.action_space.sample())

        new_value = getter(env.unwrapped)

        assert np.allclose(new_value, expected_value), \
            f"Parameter '{param_name}' in '{env_id}' was not updated correctly. " \
            f"Expected: {expected_value}, Got: {new_value}"
                
        env.close()
        ns_env.close()


@pytest.mark.parametrize("env_id", SUPPORTED_MUJOCO_ENV_IDS)
def test_mujoco_wrapper_reset(env_id):

    if env_id == "Humanoid-v5" or env_id == "HumanoidStandup-v5":
        pytest.skip("Known stability issues with Humanoid-v5 and HumanoidStandup-v5 reset.")

    #grab initial param values

    from ns_gym.wrappers.mujoco_env import MujocoWrapper, param_look_up
    from ns_gym.base import TUNABLE_PARAMS
    assert MujocoWrapper is not None
    assert param_look_up is not None
    assert TUNABLE_PARAMS is not None
    

    env = gym.make(env_id)

    env_name = env.unwrapped.__class__.__name__
    tunable_params = list(ns_gym.base.TUNABLE_PARAMS[env_name].keys())

    param_map = {param: ns_gym.update_functions.IncrementUpdate(ns_gym.schedulers.ContinuousScheduler(), k=0.1) for param in tunable_params}

    ns_env = MujocoWrapper(env, param_map, change_notification=True, delta_change_notification=True)
    obs,info = ns_env.reset(seed=42)

    init_state = obs['state']
    init_rel_time = obs['relative_time']

    # take a few steps to change parameters
    for _ in range(2):
        obs,reward,done,truncated,info = ns_env.step(ns_env.action_space.sample())
        if done:
            obs, info = ns_env.reset(seed=42)
            init_state = obs['state']
            init_rel_time = obs['relative_time']
    # check that parameters have changed

    for param in tunable_params:
        getter, _ = param_look_up(env_name, param)[0]
        current_value = getter(env.unwrapped)
        initial_value = TUNABLE_PARAMS[env_name][param]

        if param == "gravity":
            assert not np.allclose(current_value, initial_value, atol=1e-6), f"Parameter '{param}' did not change after steps."

        else:
            assert not np.allclose(current_value, initial_value, atol=1e-6), f"Parameter '{param}' did not change after steps."


    obs, info = ns_env.reset(seed=42)


    if env_id == "Humanoid-v5":
        print("STATE AFTER RESET:", obs['state'])
        print("INITIAL STATE:", init_state)

    assert np.allclose(obs['state'], init_state,   atol=1e-6), "State did not reset to initial state."
    assert obs['relative_time'] == init_rel_time, "Relative time did not reset to initial relative time."

    # check that parameters have reset

    for param in tunable_params:
        getter, _ = param_look_up(env_name, param)[0]
        current_value = getter(env.unwrapped)
        initial_value = TUNABLE_PARAMS[env_name][param]

        if param == "gravity":
            # Special case for gravity, which is a 3D vector in MuJoC
            assert np.allclose(current_value, initial_value, atol=1e-6), f"Parameter '{param}' did not reset to initial value."
        else:
            assert np.allclose(current_value, initial_value, atol=1e-6), f"Parameter '{param}' did not reset to initial value."

@pytest.mark.parametrize("env_id", SUPPORTED_MUJOCO_ENV_IDS)
def test_wrapped_vs_unwrapped_dynamics_divergence(env_id):
    """
    Tests that the MujocoWrapper correctly induces dynamic changes by running a short
    rollout to let small differences accumulate.
    """

    from ns_gym.wrappers.mujoco_env import MujocoWrapper, param_look_up
    from ns_gym.base import TUNABLE_PARAMS
    assert MujocoWrapper is not None
    assert param_look_up is not None
    assert TUNABLE_PARAMS is not None
    
    env_name = gym.make(env_id).unwrapped.__class__.__name__
    all_param_names = list(TUNABLE_PARAMS[env_name].keys())
    
    # Let the simulation run for a few steps to amplify dynamic differences
    num_steps = 100
    
    for param_name in all_param_names:
        unwrapped_env = gym.make(env_id)
        
        env_no_change = MujocoWrapper(
            gym.make(env_id),
            tunable_params={param_name: IncrementUpdate(ContinuousScheduler(start=0), k=0.0)}
        )
        env_with_change = MujocoWrapper(
            gym.make(env_id),
            tunable_params={param_name: IncrementUpdate(ContinuousScheduler(start=0), k=0.523)}
        )

        # --- Generate a consistent action sequence ---
        seed = 42
        unwrapped_env.reset(seed=seed)
        actions = [unwrapped_env.action_space.sample() for _ in range(num_steps)]

        # action_shape = unwrapped_env.action_space.shape
        # # A constant action of `1.0` will push the agent and reveal friction/gravity effects.
        # constant_action = np.ones(action_shape, dtype=unwrapped_env.action_space.dtype)
        # actions = [constant_action for _ in range(num_steps)]

        # --- 1. Test for IDENTICAL behavior with ZERO increment ---
        obs_unwrapped, _ = unwrapped_env.reset(seed=seed)
        obs_wrapped_no_change, _ = env_no_change.reset(seed=seed)
        
        for i in range(num_steps):
            obs_unwrapped, rew_unwrapped, done_unwrapped, terminated_unwrapped, info_unwrapped = unwrapped_env.step(actions[i])
            obs_wrapped_no_change, rew_wrapped_no_change, done_no_change, terminated_no_change, info_no_change = env_no_change.step(actions[i])

            if done_unwrapped or done_no_change:
                break

        assert np.allclose(obs_wrapped_no_change['state'], obs_unwrapped, atol=1e-6), \
            f"[{env_id}/{param_name}] Final states diverged after {num_steps} steps with k=0. Expected identical dynamics."

        # --- 2. Test for DIVERGENT behavior with NON-ZERO increment ---
        unwrapped_env.reset(seed=seed) # Reset to run again for a fair comparison
        obs_wrapped_with_change, _ = env_with_change.reset(seed=seed)

        for i in range(num_steps):
            obs_unwrapped, rew_unwrapped, done_unwrapped, terminated_unwrapped, info_unwrapped = unwrapped_env.step(actions[i])
            obs_wrapped_with_change, rew_wrapped_with_change, done_wrapped_with_change, terminated_wrapped_with_change, info_wrapped_with_change = env_with_change.step(actions[i])

            if done_unwrapped or done_wrapped_with_change:
                break

        assert not np.allclose(obs_wrapped_with_change['state'], obs_unwrapped, atol=1e-6), \
            f"[{env_id}/{param_name}] Final states did NOT diverge after {num_steps} steps with k=0.523. Wrapper failed to change dynamics."

        # --- Cleanup ---
        unwrapped_env.close()
        env_no_change.close()
        env_with_change.close()