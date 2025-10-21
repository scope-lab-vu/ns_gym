import pytest
import ns_gym
import gymnasium as gym
import numpy as np
from ns_gym.envs.vehicle_tracking import Actions

# --- Fixtures ---

@pytest.fixture
def single_pursuer_env():
    """Provides a default 5x5 environment with one pursuer."""
    return gym.make('ns_gym/VehicleTracking-v0', obstacle_map=np.zeros((5, 5)))

@pytest.fixture
def multi_pursuer_env():
    """Provides a 10x10 environment with two pursuers."""
    return gym.make(
        'ns_gym/VehicleTracking-v0',
        num_pursuers=2,
        obstacle_map=np.zeros((10, 10), dtype=bool),    
        starting_pursuer_position=[(0, 0), (9, 9)],
        starting_evader_position=(5, 5),
        goal_locations=[(0, 9), (9, 0)],
        fov_distance=20,
        fov_angle=2*np.pi/180,
    )

def test_make_single_pursuer_env(single_pursuer_env):
    """Tests if a single-pursuer environment is created with correct defaults."""
    env = single_pursuer_env
    assert env is not None
    assert isinstance(env, gym.Env)
    assert env.unwrapped.num_pursuers == 1
    
def test_make_multi_pursuer_env(multi_pursuer_env):
    """Tests if a multi-pursuer environment is created successfully."""
    env = multi_pursuer_env
    assert env is not None
    assert env.unwrapped.num_pursuers == 2
    assert np.array_equiv(env.unwrapped.starting_pursuer_positions[0], np.array([0, 0]))
    assert np.array_equiv(env.unwrapped.starting_pursuer_positions[1], np.array([9, 9]))
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert env.action_space.shape == (2,)

def test_reset_multi_pursuer(multi_pursuer_env):
    """Tests the reset method for a multi-pursuer environment."""
    env = multi_pursuer_env
    obs, info = env.reset()

    assert "pursuer_position" in obs
    assert "evader_position" in obs
    assert "goal_position" in obs
    assert isinstance(obs["pursuer_position"], np.ndarray)
    assert obs["pursuer_position"].shape == (2,)

    start_pursuer1_flat = env.unwrapped._to_flattened(env.unwrapped.starting_pursuer_positions[0])
    start_pursuer2_flat = env.unwrapped._to_flattened(env.unwrapped.starting_pursuer_positions[1])

    assert np.array_equiv(obs["pursuer_position"], [start_pursuer1_flat, start_pursuer2_flat])

    assert np.array_equiv(env.unwrapped.pursuer_positions, env.unwrapped.starting_pursuer_positions)


def test_check_fov_multi_pursuer(multi_pursuer_env):
    """Tests if _evader_inview works correctly with multiple pursuers.
    
    P . . . . . . . . G
    . . . . . . . . . .
    . . . . . . . . . .
    . . . . . . . . . .
    . . . . . . . . . .
    . . . . . E . . . .
    . . . . . . . . . .
    . . . . . . . . . .
    . . . . . . . . . .
    G . . . . . . . . P

    """
    
    env = multi_pursuer_env
    env.reset()

    assert np.array_equiv(env.unwrapped.pursuer_positions, [[0, 0], [9, 9]])
    assert np.array_equiv(env.unwrapped.pursuer_dirs, [[1, 0], [1, 0]]) 
    assert env.unwrapped._evader_inview() is False, "Should be False if any pursuer has line of sight"


    env.step([Actions.STAY.value, Actions.ROTATE_CCW.value])
    env.step([Actions.STAY.value, Actions.ROTATE_CCW.value])
    env.step([Actions.STAY.value, Actions.ROTATE_CCW.value])
    env.step([Actions.STAY.value, Actions.ROTATE_CCW.value])
    env.step([Actions.STAY.value, Actions.STAY.value])

    max_steps = 20
    in_view = False
    for _ in range(max_steps):
        obs, reward, done, truncated, info = env.step([Actions.STAY.value, Actions.STAY.value])
        if env.unwrapped._evader_inview():
            in_view = True
            break

    assert in_view is True, "Evader should eventually come into view because we are looking at goal location"



def test_pursuer_step_multi_pursuer(multi_pursuer_env):
    """Tests a single step in a multi-pursuer environment."""
    env = multi_pursuer_env
    env.reset()

    # Define actions for both pursuers
    actions = [Actions.RIGHT.value, Actions.LEFT.value]
    obs, reward, done, truncated, info = env.step(actions)

    # Pursuer 1 starts at (0,0), moves right to (1,0)
    expected_p1_pos_flat = env.unwrapped._to_flattened(np.array([1, 0]))
    # Pursuer 2 starts at (9,9), moves left to (8,9)
    expected_p2_pos_flat = env.unwrapped._to_flattened(np.array([8, 9]))

    expected_pursuer_positions = np.array([expected_p1_pos_flat, expected_p2_pos_flat])
    assert np.array_equiv(obs["pursuer_position"], expected_pursuer_positions), "Pursuers did not move as expected"

    # Test capture condition


    



def test_flatten_unflatten(single_pursuer_env):
    """Tests the flattening and unflattening of grid positions."""
    env = single_pursuer_env
    env.reset()

    for r in range(env.unwrapped.map_dimensions[1]):
        for c in range(env.unwrapped.map_dimensions[0]):
            pos = np.array([c, r])
            flat = env.unwrapped._to_flattened(pos)
            unflat = env.unwrapped._from_flattened(flat)
            assert np.array_equiv(unflat, pos), f"Flatten/Unflatten failed for position {pos}"