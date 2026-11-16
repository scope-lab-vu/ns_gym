"""``ns_gym.utils``: the distance used for ``delta_change`` and small helpers."""

import numpy as np
import pytest

from ns_gym import utils
from ns_gym.base import Reward


# ============================================================
# wasserstein_distance: the size of every distribution change
# ============================================================

@pytest.mark.parametrize(
    "u,v,expected",
    [
        ([1, 0, 0], [1, 0, 0], 0.0),
        ([1, 0, 0], [0, 1, 0], 1.0),
        ([1, 0, 0], [0, 0, 1], 2.0),  # all mass moved two indices
        ([1, 0, 0], [0.5, 0.5, 0], 0.5),
        ([1, 0, 0], [0.9, 0.05, 0.05], 0.15),
        ([0.25, 0.25, 0.25, 0.25], [0, 0, 0, 1], 1.5),
    ],
)
def test_wasserstein_distance_known_values(u, v, expected):
    assert np.isclose(utils.wasserstein_distance(u, v), expected)


def test_wasserstein_distance_is_symmetric():
    u, v = [0.7, 0.2, 0.1], [0.1, 0.3, 0.6]
    assert np.isclose(utils.wasserstein_distance(u, v), utils.wasserstein_distance(v, u))


def test_wasserstein_distance_satisfies_triangle_inequality():
    a, b, c = [1, 0, 0], [0.2, 0.5, 0.3], [0, 0, 1]
    d = utils.wasserstein_distance
    assert d(a, c) <= d(a, b) + d(b, c) + 1e-12


def test_wasserstein_distance_returns_python_float():
    assert type(utils.wasserstein_distance([1, 0], [0, 1])) is float


def test_wasserstein_distance_accepts_arrays():
    assert np.isclose(utils.wasserstein_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 1.0)


def test_wasserstein_distance_rejects_different_lengths():
    with pytest.raises(ValueError):
        utils.wasserstein_distance([1, 0, 0], [1, 0])


def test_wasserstein_distance_rejects_negative_weights():
    with pytest.raises(ValueError):
        utils.wasserstein_distance([1, 0, 0], [-0.1, 0.55, 0.55])


# ============================================================
# Helpers
# ============================================================

def test_categorical_sample_follows_a_degenerate_distribution():
    assert all(utils.categorical_sample([0.0, 1.0, 0.0]) == 1 for _ in range(20))


def test_categorical_sample_stays_in_range():
    samples = {int(utils.categorical_sample([0.2, 0.3, 0.5])) for _ in range(200)}
    assert samples <= {0, 1, 2}
    assert len(samples) > 1


def test_state_action_update_replaces_only_probabilities():
    transitions = [(1.0, 4, 0.0, False), (0.0, 1, 0.0, False), (0.0, 0, 1.0, True)]
    updated = utils.state_action_update(transitions, [0.8, 0.1, 0.1])
    assert updated == [(0.8, 4, 0.0, False), (0.1, 1, 0.0, False), (0.1, 0, 1.0, True)]


@pytest.mark.parametrize("n,k,expected", [(5, 2, 10), (5, 0, 1), (5, 5, 1), (10, 3, 120)])
def test_n_choose_k(n, k, expected):
    assert utils.n_choose_k(n, k) == expected


def test_type_mismatch_checker_unwraps_ns_gym_types():
    obs = {"state": 3, "env_change": {"P": 0}, "delta_change": {"P": 0.0}, "relative_time": 1}
    reward = Reward(reward=1.5, env_change={"P": 0}, delta_change={"P": 0.0}, relative_time=1)
    assert utils.type_mismatch_checker(obs, reward) == (3, 1.5)


def test_type_mismatch_checker_passes_plain_values_through():
    state = np.array([0.1, 0.2])
    out_state, out_reward = utils.type_mismatch_checker(state, 2.0)
    assert out_state is state
    assert out_reward == 2.0
    assert utils.type_mismatch_checker() == (None, None)
