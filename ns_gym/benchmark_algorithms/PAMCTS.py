"""Policy-Augmented MCTS (PA-MCTS).

Combines a learned action-value head (DDQN, SB3 DQN, or any callable
that returns per-action Q-values for a given (state, env)) with online
MCTS via a convex combination after both heads are normalized to
``[0, 1]``.

Two ways to wire up the Q-head:

1. **Legacy** -- pass ``DDQN_model`` (a torch ``nn.Module`` instance)
   and ``DDQN_model_path`` (a state-dict ``.pth``). PAMCTS builds an
   internal :class:`~ns_gym.benchmark_algorithms.DDQN.DDQN.DQNAgent`
   and feeds it the raw state.

2. **Recommended** -- pass ``q_value_fn``, a callable
   ``q_value_fn(state, env) -> np.ndarray`` that returns the per-action
   Q-values. This decouples PAMCTS from the ns_gym DQN format and lets
   you plug in any value head: a Stable-Baselines3 DQN via
   :class:`~ns_gym.base.StableBaselineWrapper`, a hand-trained PyTorch
   net, even a precomputed table.

Exactly one of those two paths must be provided.
"""

import numpy as np

import ns_gym as nsg
from ns_gym import base


class PAMCTS(base.Agent):
    """Policy-Augmented MCTS.

    Args:
        alpha (float): Convex combination weight on the learned
            policy's Q-values. ``alpha = 0`` -> pure MCTS,
            ``alpha = 1`` -> pure learned head.
        mcts_iter (int): Total MCTS rollouts (m).
        mcts_search_depth (int): MCTS rollout depth (d).
        mcts_discount_factor (float): MCTS discount factor (gamma).
        mcts_exploration_constant (float): UCB1 exploration constant (c).
        state_space_size (int): Discrete state-space size, only used
            by the legacy DDQN wiring path.
        action_space_size (int): Number of discrete actions, only used
            by the legacy DDQN wiring path.
        DDQN_model (torch.nn.Module, optional): Legacy. Architecture
            instance whose weights will be loaded from
            ``DDQN_model_path``.
        DDQN_model_path (str, optional): Legacy. Path to a state-dict
            ``.pth`` file matching ``DDQN_model``.
        q_value_fn (Callable[[state, env], np.ndarray], optional):
            Recommended. A function that returns per-action Q-values
            for a given ``(state, env)`` pair. Bypasses the legacy
            DDQN wiring entirely.
        seed (int): Random seed forwarded to the legacy
            :class:`DQNAgent`. Unused when ``q_value_fn`` is supplied.

    Examples:
        >>> # Legacy: ns_gym DQN architecture + .pth state dict
        >>> from ns_gym.benchmark_algorithms.DDQN.DDQN import DQN as DDQNNet
        >>> arch = DDQNNet(state_size=16, action_size=4,
        ...                num_layers=3, num_hidden_units=64, seed=0)
        >>> agent = PAMCTS(alpha=0.75, mcts_iter=30, mcts_search_depth=20,
        ...                mcts_discount_factor=0.95,
        ...                mcts_exploration_constant=1.4,
        ...                state_space_size=16, action_space_size=4,
        ...                DDQN_model=arch,
        ...                DDQN_model_path="weights.pth")

        >>> # Recommended: Stable-Baselines3 DQN via StableBaselineWrapper
        >>> from stable_baselines3 import DQN
        >>> from ns_gym.base import StableBaselineWrapper
        >>> sb3 = DQN.load("contextual_ddqn.zip")
        >>> def obs_fn(state, env):
        ...     return np.concatenate([
        ...         np.eye(16, dtype=np.float32)[int(state)],
        ...         np.asarray(env.transition_prob, dtype=np.float32),
        ...     ])
        >>> wrap = StableBaselineWrapper(sb3, obs_fn=obs_fn)
        >>> agent = PAMCTS(alpha=0.75, mcts_iter=30, mcts_search_depth=20,
        ...                mcts_discount_factor=0.95,
        ...                mcts_exploration_constant=1.4,
        ...                state_space_size=16, action_space_size=4,
        ...                q_value_fn=wrap.q_values)
    """

    def __init__(self,
                 alpha,
                 mcts_iter,
                 mcts_search_depth,
                 mcts_discount_factor,
                 mcts_exploration_constant,
                 state_space_size,
                 action_space_size,
                 DDQN_model=None,
                 DDQN_model_path=None,
                 q_value_fn=None,
                 seed=0) -> None:
        self.alpha = alpha
        self.mcts_iter = mcts_iter
        self.mcts_search_depth = mcts_search_depth
        self.mcts_discount_factor = mcts_discount_factor
        self.mcts_exploration_constant = mcts_exploration_constant
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        # Wire the Q-head exactly once.
        if q_value_fn is not None:
            if DDQN_model is not None or DDQN_model_path is not None:
                raise ValueError(
                    "PAMCTS: pass *either* q_value_fn or the legacy "
                    "(DDQN_model, DDQN_model_path) pair -- not both."
                )
            if not callable(q_value_fn):
                raise TypeError(
                    "PAMCTS: q_value_fn must be a callable "
                    "f(state, env) -> np.ndarray of per-action Q-values."
                )
            self._q_value_fn = q_value_fn
            self.DDQN_agent = None
        else:
            if DDQN_model is None or DDQN_model_path is None:
                raise ValueError(
                    "PAMCTS: provide either q_value_fn (recommended) or "
                    "both DDQN_model and DDQN_model_path (legacy)."
                )
            self.DDQN_model = DDQN_model
            self.DDQN_model_path = DDQN_model_path
            self.DDQN_agent = nsg.benchmark_algorithms.DDQN.DQNAgent(
                self.state_space_size,
                self.action_space_size,
                seed=seed,
                model=self.DDQN_model,
                model_path=self.DDQN_model_path,
            )
            self._q_value_fn = None

    def _learned_q(self, state, env):
        """Per-action Q-values from whichever wiring was selected."""
        if self._q_value_fn is not None:
            return np.asarray(self._q_value_fn(state, env), dtype=np.float32)
        # Legacy DDQN_agent.search returns (action, all_q_values)
        _, q_values = self.DDQN_agent.search(state)
        return np.asarray(q_values, dtype=np.float32).ravel()

    def search(self, state, env, normalize=True):
        self.mcts_agent = nsg.benchmark_algorithms.MCTS(
            env, state,
            d=self.mcts_search_depth,
            m=self.mcts_iter,
            c=self.mcts_exploration_constant,
            gamma=self.mcts_discount_factor,
        )
        _mcts_action, mcts_action_values = self.mcts_agent.search()
        ddqn_action_values = self._learned_q(state, env)
        mcts_action_values = np.asarray(mcts_action_values, dtype=np.float32)

        if normalize:
            eps = 1e-8
            ddqn_action_values = (
                (ddqn_action_values - ddqn_action_values.min())
                / (ddqn_action_values.max() - ddqn_action_values.min() + eps)
            )
            mcts_action_values = (
                (mcts_action_values - mcts_action_values.min())
                / (mcts_action_values.max() - mcts_action_values.min() + eps)
            )

        hybrid = self._get_pa_uct_score(
            self.alpha, ddqn_action_values, mcts_action_values,
        )
        return int(np.argmax(hybrid)), hybrid

    def act(self, state, env, normalize=True):
        action, _ = self.search(state, env, normalize)
        return action

    def _get_pa_uct_score(self, alpha, policy_value, mcts_return):
        return (alpha * policy_value) + ((1.0 - alpha) * mcts_return)


if __name__ == "__main__":
    pass
