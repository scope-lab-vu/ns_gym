import pytest
import numpy as np
import gymnasium as gym
import CityEnvGym

@pytest.fixture
def env():
    """Pytest fixture to create a CityEnv instance for testing."""

    target_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    drone_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    sensors = [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]] # x,y ,radius
    env = gym.make("CityEnvGym/CityEnv-v0", render_mode="human",sensors=sensors,num_evader_steps=50,max_episode_steps=18000, time_step=1/60.0, fov_angle=90.0, fov_distance=100.0,target_physics=target_physics, drone_physics=drone_physics)
    return env


def test_load_pursuit_evasion_env():
    import gymnasium as gym
    import CityEnvGym
    import ns_gym

    
    target_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    drone_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    sensors = [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]] # x,y ,radius
    env = gym.make("CityEnvGym/CityEnv-v0", render_mode="human",sensors=sensors,num_evader_steps=50,max_episode_steps=18000, time_step=1/60.0, fov_angle=90.0, fov_distance=100.0,target_physics=target_physics, drone_physics=drone_physics)

    assert env is not None


def test_wrap_pursuit_evasion_env(env):
    from ns_gym.wrappers import PursuitEvasionWrapper
    assert PursuitEvasionWrapper is not None
    ns_env = PursuitEvasionWrapper(env, tunable_params={"drone_physics.max_speed": float, "target_physics.max_speed": float}, change_notification=True, delta_change_notification=True, in_sim_change=True)


def test_get_planning_env():
    pass


def test_reset():
    pass


def test_step():
    pass