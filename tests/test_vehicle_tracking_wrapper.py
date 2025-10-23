import pytest 




VEHICLE_TRACKING_PARAMS = ["num_pursuers", 
                           "fov_distance", 
                           "fov_angle",
                           "game_mode",
                           "is_evader_always_observable",
                           "allow_diagonal_movement",
                           "goal_locations"
                       ]





def test_vehicle_tracking_wrapper():
    from ns_gym.envs.vehicle_tracking import VehicleTrackingWrapper
    import gymnasium as gym
    import numpy as np

    env = gym.make("ns_gym/VehicleTracking-v0")
    wrapped_env = VehicleTrackingWrapper(env)

    obs = wrapped_env.reset()
    assert obs.shape == (10, 10, 3), "Observation shape should be (10, 10, 3)"

    action = np.array([0.0, 1.0])  # Move up
    obs, reward, done, info = wrapped_env.step(action)
    assert obs.shape == (10, 10, 3), "Observation shape should be (10, 10, 3) after step"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(done, bool), "Done should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"


def test_vehicle_tracking_wrapper_params():
    pass


def test_deepcopy():
    pass

def test_get_planning_env():
    pass