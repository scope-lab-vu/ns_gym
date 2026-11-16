"""MCTS and PA-MCTS: that they run against ns_gym planning envs and that their
value estimates mean what they say."""

import random

import numpy as np
import pytest

from ns_gym.benchmark_algorithms import MCTS, PAMCTS
from ns_gym.benchmark_algorithms.MCTS import ChanceNode, DecisionNode

from _helpers import CASE_BY_ID, known_bug

RIGHT = 2


@pytest.fixture
def frozen_lake():
    """Deterministic 4x4 FrozenLake at its start state, and the reset observation."""
    case = CASE_BY_ID["FrozenLake-v1"]
    env = case.make(case.no_update_fn())
    obs, _ = env.reset(seed=0)
    return env, obs


@pytest.fixture(autouse=True)
def _seed_global_rngs():
    """MCTS draws from the global ``random`` and ``numpy.random`` generators."""
    random.seed(0)
    np.random.seed(0)


# ============================================================
# MCTS
# ============================================================

def test_mcts_search_returns_an_action_and_one_value_per_action(frozen_lake):
    env, obs = frozen_lake
    action, q_values = MCTS(env.get_planning_env(), obs, d=20, m=50, c=1.4, gamma=0.95).search()
    assert int(action) in range(4)
    assert len(q_values) == 4


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "path,simulations",
    [([1, 1, 2, 1, 2], 50), ([1, 1, 2, 1], 100)],
    ids=["one-step-from-goal", "two-steps-from-goal"],
)
def test_mcts_takes_the_rewarding_action_near_the_goal(frozen_lake, path, simulations, seed):
    """On the non-slippery lake, with the goal one or two moves to the right,
    MCTS moves right. (From the start state a few hundred simulations of
    uniform rollouts do no better than chance, so that is not asserted.)"""
    random.seed(seed)
    np.random.seed(seed)
    env, obs = frozen_lake
    for action in path:
        obs, _, _, _, _ = env.step(action)

    action, q_values = MCTS(env.get_planning_env(), obs, d=20, m=simulations, c=1.4, gamma=0.95).search()
    assert int(action) == RIGHT
    assert q_values[RIGHT] == max(q_values) > 0


def test_mcts_does_not_advance_the_real_env(frozen_lake):
    env, obs = frozen_lake
    MCTS(env.get_planning_env(), obs, d=20, m=50, c=1.4, gamma=0.95).search()
    assert env.t == 0
    assert env.unwrapped.s == obs["state"]


def test_mcts_act_accepts_the_observation_dict(frozen_lake):
    env, obs = frozen_lake
    agent = MCTS(env.get_planning_env(), obs, d=20, m=50, c=1.4, gamma=0.95)
    assert int(agent.act(obs, env.get_planning_env())) in range(4)


def test_mcts_rejects_continuous_action_spaces():
    case = CASE_BY_ID["Pendulum-v1"]
    env = case.make()
    obs, _ = env.reset(seed=0)
    with pytest.raises(ValueError):
        MCTS(env, obs, d=5, m=5, c=1.4, gamma=0.95)


def test_mcts_search_with_fewer_simulations_than_actions(frozen_lake):
    env, obs = frozen_lake
    action, _ = MCTS(env.get_planning_env(), obs, d=5, m=2, c=1.4, gamma=0.95).search()
    assert int(action) in range(4)


@known_bug("F1")
def test_mcts_backpropagation_discounts_once_per_decision(frozen_lake):
    """A return of 1 found three decisions below the root is worth 1, gamma and
    gamma**2 to the three actions on the path."""
    env, obs = frozen_lake
    gamma = 0.9
    agent = MCTS(env.get_planning_env(), obs, d=5, m=5, c=1.4, gamma=gamma)

    root = agent.v0
    chance_0 = ChanceNode(root, 0)
    decision_1 = DecisionNode(chance_0, 101, 1, False, 0)
    chance_1 = ChanceNode(decision_1, 0)
    decision_2 = DecisionNode(chance_1, 102, 1, False, 0)
    chance_2 = ChanceNode(decision_2, 0)
    leaf = DecisionNode(chance_2, 103, 1, False, 0)

    agent._backpropagation(1.0, leaf)

    assert np.isclose(agent.Qsa[(102, 0)], 1.0)
    assert np.isclose(agent.Qsa[(101, 0)], gamma)
    assert np.isclose(agent.Qsa[(root.state, 0)], gamma**2)


# ============================================================
# PA-MCTS
# ============================================================

def _pamcts(**overrides):
    kwargs = dict(
        alpha=1.0,
        mcts_iter=30,
        mcts_search_depth=10,
        mcts_discount_factor=0.95,
        mcts_exploration_constant=1.4,
        state_space_size=16,
        action_space_size=4,
    )
    kwargs.update(overrides)
    return PAMCTS(**kwargs)


def test_pamcts_with_alpha_one_follows_the_learned_values(frozen_lake):
    env, obs = frozen_lake
    agent = _pamcts(alpha=1.0, q_value_fn=lambda state, env: np.array([0.0, 0.1, 5.0, 0.2]))
    action, scores = agent.search(obs, env.get_planning_env())
    assert action == 2
    assert len(scores) == 4
    assert agent.act(obs, env.get_planning_env()) == 2


def test_pamcts_scores_are_a_convex_combination(frozen_lake):
    env, obs = frozen_lake
    agent = _pamcts(alpha=0.5, q_value_fn=lambda state, env: np.array([0.0, 1.0, 2.0, 3.0]))
    _, scores = agent.search(obs, env.get_planning_env())
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0 + 1e-6)


def test_pamcts_requires_exactly_one_value_head():
    with pytest.raises(ValueError):
        _pamcts()
    with pytest.raises(ValueError):
        _pamcts(q_value_fn=lambda s, e: np.zeros(4), DDQN_model=object(), DDQN_model_path="weights.pth")
    with pytest.raises(TypeError):
        _pamcts(q_value_fn="not callable")
