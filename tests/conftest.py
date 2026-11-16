"""Fixtures shared by the whole test suite.

Constants, the env registry and helper functions live in ``_helpers.py`` so
test modules can import them by name; this file holds only fixtures.
"""

import pytest

from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import DistributionDecrementUpdate, IncrementUpdate

from _helpers import GRIDWORLD_WRAPPERS


@pytest.fixture
def cc_params():
    """Two tunable parameters per classic-control env, updated on every step.

    Increments are sized to the parameter. MountainCar's gravity and force are
    of order 1e-3, so a step of 0.1 would either be rejected by the constraint
    checker or saturate the velocity clip, and the env would look stationary.
    """
    fn = IncrementUpdate(ContinuousScheduler(), k=0.1)
    return {
        "CartPole-v1": {"masspole": fn, "gravity": fn},
        "Acrobot-v1": {"LINK_LENGTH_1": fn, "LINK_MASS_2": fn},
        "MountainCar-v0": {
            "gravity": IncrementUpdate(ContinuousScheduler(), k=0.0005),
            "force": IncrementUpdate(ContinuousScheduler(), k=0.0002),
        },
        "MountainCarContinuous-v0": {
            "power": IncrementUpdate(ContinuousScheduler(), k=0.0003),
        },
        "Pendulum-v1": {"m": fn, "g": fn},
    }


@pytest.fixture
def gw_wrappers():
    """Env id -> wrapper class for the gymnasium gridworlds."""
    return dict(GRIDWORLD_WRAPPERS)


@pytest.fixture
def gw_params():
    """Transition-probability update for the gridworlds, on every step.

    ``DistributionDecrementUpdate`` clamps at 0, so the distribution stays
    valid however long an episode runs.
    """
    fn = DistributionDecrementUpdate(ContinuousScheduler(), k=0.1)
    return {
        "CliffWalking-v1": {"P": fn},
        "FrozenLake-v1": {"P": fn},
    }


@pytest.fixture
def mj_params():
    """A representative subset of tunable parameters for each MuJoCo env."""
    fn = IncrementUpdate(ContinuousScheduler(), k=0.1)
    return {
        "Ant-v5": {"torso_mass": fn},
        "HalfCheetah-v5": {"torso_mass": fn, "floor_friction": fn},
        "Hopper-v5": {"torso_mass": fn, "thigh_mass": fn},
        "Humanoid-v5": {"torso_mass": fn},
        "HumanoidStandup-v5": {"torso_mass": fn},
        "InvertedPendulum-v5": {"pole_mass": fn, "cart_mass": fn},
        "InvertedDoublePendulum-v5": {"cart_mass": fn, "pole1_mass": fn},
        "Reacher-v5": {"body0_mass": fn, "body1_mass": fn},
        "Swimmer-v5": {"body_mid_mass": fn},
        "Pusher-v5": {"r_shoulder_pan_link_mass": fn},
    }
