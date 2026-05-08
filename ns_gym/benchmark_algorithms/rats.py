"""
Risk Averse Tree Search (RATS) algorithm.

Reference: Lecarpentier and Rachelson, "Non-Stationary Markov Decision
Processes: a Worst-Case Approach using Model-Based Reinforcement Learning",
NeurIPS 2019.

Compatible with NS-Gym discrete environments that expose a transition
table on the unwrapped env. The agent reads the model directly from the
env -- it does not call env.step -- so the env must provide:

    env.unwrapped.s         - current discrete state
    env.unwrapped.P[s][a]   - list of (prob, next_state, reward, done) tuples
    env.action_space        - gymnasium.spaces.Discrete

This contract is satisfied by ns_gym.wrappers.NSFrozenLakeWrapper,
ns_gym.wrappers.NSCliffWalkingWrapper, and the ns_gym Bridge env.
"""

import numpy as np
from copy import deepcopy
from gymnasium import spaces

import ns_gym.base as base


def _assert_types(p, types_list):
    assert len(p) == len(types_list), \
        'Error: expected {} parameters received {}'.format(len(types_list), len(p))
    for i, item in enumerate(p):
        assert type(item) == types_list[i], \
            'Error: wrong type, expected {}, received {}'.format(types_list[i], type(item))


def _node_value(node):
    assert node.value is not None, 'Error: node value={}'.format(node.value)
    return node.value


def _get_state(env):
    return getattr(env, 'unwrapped', env).s


def _get_P(env):
    return getattr(env, 'unwrapped', env).P


class DecisionNode:
    """Decision node, labelled by a state."""
    def __init__(self, parent, state, weight, is_terminal, reward=0.0):
        self.parent = parent
        self.state = state
        self.weight = weight
        self.initial_weight = weight
        self.is_terminal = is_terminal
        self.reward = reward
        self.depth = 0 if parent is None else parent.depth + 1
        self.children = []
        self.value = None


class ChanceNode:
    """Chance node, labelled by a state-action pair (state via parent)."""
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []
        self.value = None


class RATS(base.Agent):
    """Risk-Averse Tree Search agent.

    Args:
        action_space (gymnasium.spaces.Discrete): action space of the env.
        gamma (float): discount factor.
        max_depth (int): planning depth of the minimax tree.
        L_p (float): Lipschitz constant of the transition kernel in time.
        L_r (float): Lipschitz constant of the reward in time.
        tau (float): non-stationarity time scale.
    """

    def __init__(self, action_space, gamma=0.9, max_depth=4,
                 L_p=1.0, L_r=0.0, tau=1.0):
        super().__init__()
        self.action_space = action_space
        self.n_actions = action_space.n
        self.gamma = gamma
        self.max_depth = max_depth
        self.L_p = L_p
        self.L_r = L_r
        self.tau = tau
        self.t_call = 0

    def reset(self, p=None):
        if p is None:
            self.__init__(self.action_space)
        else:
            _assert_types(p, [spaces.Discrete, float, int])
            self.__init__(p[0], p[1], p[2])

    def display(self):
        print('Displaying RATS agent:')
        print('Action space     :', self.action_space)
        print('Number of actions:', self.n_actions)
        print('Gamma            :', self.gamma)
        print('Maximum depth    :', self.max_depth)

    def build_tree(self, node, env):
        P = _get_P(env)
        if isinstance(node, DecisionNode):
            if node.depth < self.max_depth:
                for a in range(self.n_actions):
                    node.children.append(ChanceNode(node, a))
            else:
                return None
        else:  # ChanceNode
            for prob, next_state, reward, done in P[node.parent.state][node.action]:
                node.children.append(
                    DecisionNode(
                        parent=node,
                        state=next_state,
                        weight=prob,
                        is_terminal=bool(done),
                        reward=float(reward),
                    )
                )
        for ch in node.children:
            if isinstance(ch, DecisionNode):
                if not ch.is_terminal:
                    self.build_tree(ch, env)
            else:
                self.build_tree(ch, env)

    def initialize_tree(self, env, done):
        root = DecisionNode(None, _get_state(env), 1.0, done)
        self.build_tree(root, env)
        return root

    def minimax(self, node, env):
        if isinstance(node, DecisionNode):
            assert node.value is None, 'Error: node value={}'.format(node.value)
            if node.depth == self.max_depth:
                node.value = self.heuristic_value(node, env)
            elif node.is_terminal:
                node.value = node.reward
            else:
                v = -np.inf
                for ch in node.children:
                    v = max(v, self.minimax(ch, env))
                node.value = v
        else:  # ChanceNode
            self.set_worst_case_distribution(node, env)
            v = 0.0
            for ch in node.children:
                v += ch.weight * ch.value
            v *= self.gamma
            R = 0.0
            for ch in node.children:
                R += ch.reward * ch.initial_weight
            v += R - self.L_r * self.tau * node.depth
            assert node.value is None, 'Error: node value={}'.format(node.value)
            node.value = v
        return node.value

    def set_worst_case_distribution(self, node, env):
        assert isinstance(node, ChanceNode), \
            'Error: node type={}'.format(type(node))
        for ch in node.children:
            self.minimax(ch, env)
        v = np.asarray([ch.value for ch in node.children])
        w0 = np.asarray([ch.initial_weight for ch in node.children])
        c = node.depth * self.L_p * self.tau
        w = self.worstcase_distribution_direct_method(v, w0, c)
        for i, ch in enumerate(node.children):
            ch.weight = w[i]

    def worstcase_distribution_direct_method(self, v, w0, c):
        n = len(v)
        w_worst = np.zeros(n)
        w_worst[int(np.argmin(v))] = 1.0
        return w_worst

    def heuristic_value(self, node, env):
        return 0.0

    def get_depth_list(self, node, d_list):
        d_list.append(node.depth)
        for ch in node.children:
            self.get_depth_list(ch, d_list)

    def act(self, observation=None, env=None, done=False):
        """Run the RATS planning procedure and return the chosen action.

        Args:
            observation: ignored (state is read from env.unwrapped.s); kept
                for signature parity with other ns_gym agents.
            env: NS-Gym env exposing env.unwrapped.s and env.unwrapped.P.
            done (bool): True if the current state is terminal.
        """
        assert env is not None, "RATS.act requires an env argument"
        self.t_call = 0
        root = self.initialize_tree(deepcopy(env), done)
        self.minimax(root, deepcopy(env))
        return max(root.children, key=_node_value).action
