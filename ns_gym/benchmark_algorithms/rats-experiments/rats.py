"""
Risk Averse Tree Search (RATS) algorithm

Required attributes and functions of the environment:
env.state
env.tau
env.L_p
env.L_r
env.get_time()
env.is_terminal(state)
env.static_reachable_states(s, a)
env.transition_probability(s_p, s, t, a)
env.instant_reward(s, t, a, s_p)
env.expected_reward(s, t, a)
env.transition(s, a, is_model_dynamic)
"""

import numpy as np
import code.utils.utils as utils
from gym import spaces
from copy import deepcopy


def node_value(node):
    assert node.value is not None, 'Error: node value={}'.format(node.value)
    return node.value


class DecisionNode:
    """
    Decision node class, labelled by a state.
    """
    def __init__(self, parent, state, weight, is_terminal):
        self.parent = parent
        self.state = state
        self.weight = weight  # Probability to occur
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.value = None


class ChanceNode:
    """
    Chance node class, labelled by a state-action pair.
    The state is accessed via the parent attribute.
    """
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []
        self.value = None


class RATS(object):
    """
    RATS agent
    """
    def __init__(self, action_space, gamma=0.9, max_depth=4):
        self.action_space = action_space
        self.n_actions = self.action_space.n
        self.t_call = None
        self.gamma = gamma
        self.max_depth = max_depth

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        :param p: (list) list of parameters
        :return: None
        """
        if p is None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p, [spaces.discrete.Discrete, float, int])
            self.__init__(p[0], p[1], p[2])

    def display(self):
        """
        Display info about the attributes
        :return: None
        """
        print('Displaying RATS agent:')
        print('Action space       :', self.action_space)
        print('Number of actions  :', self.n_actions)
        print('Gamma              :', self.gamma)
        print('Maximum depth      :', self.max_depth)

    def build_tree(self, node, env):
        """
        Recursively build an empty tree.
        :param node: () current node
        :param env: () environment
        :return: None
        """
        if type(node) is DecisionNode:  # DecisionNode
            if node.depth < self.max_depth:
                for a in range(self.n_actions):
                    node.children.append(ChanceNode(node, a))
            else:  # Reached maximum depth
                return None
        else:  # ChanceNode
            #print("check node:",env.P[node.parent.state][node.action])
            #print(env.transition_prob)
            #ind = np.random.choice([0, 1, -1], env.transition_prob)
            reachable_states = []
            is_terminal_list = []
            for a in [node.action, (node.action+1)%4, (node.action-1)%4]:
                reachable_states.append(env.P[node.parent.state][a][0][1])
                is_terminal_list.append(env.P[node.parent.state][a][0][3])
                #print(node.parent.state, a, env.P[node.parent.state][a])
            #for s_p in env.static_reachable_states(node.parent.state, node.action):
            #print(reachable_states)
            for index, s_p in enumerate(reachable_states):
                node.children.append(
                    DecisionNode(
                        parent=node,
                        state=s_p,
                        weight=env.transition_prob[index],
                        is_terminal=is_terminal_list[index]
                    )
                )
                #print(env.transition_probability(s_p, node.parent.state, self.t_call, node.action))
        for ch in node.children:
            if type(ch) is DecisionNode:
                if not ch.is_terminal:
                    self.build_tree(ch, env)
            else:  # ChanceNode
                self.build_tree(ch, env)

    def initialize_tree(self, env, done):
        """
        Initialize an empty tree.

        The tree is composed with all the possible actions as chance nodes and all the
        possible state as decision nodes.

        The used model is the snapshot MDP provided by the environment at the time of
        the environment.

        The depth of the tree is defined by the self.max_depth attribute of the agent.

        The used heuristic for the evaluation of the leaf nodes that are not terminal
        nodes is defined by the function self.heuristic_value.

        :param env: () environment
        :param done: (bool)
        :return: (DecisionNode) root node of the created tree
        """
        root = DecisionNode(None, env.s, 1, done)
        self.build_tree(root, env)
        return root

    def minimax(self, node, env):
        """
        Recursive minimax calls.
        :param node: () current node
        :param env: () environment
        :return: value of the current node
        """
        L_r = 0
        tau = 1
        if type(node) is DecisionNode:
            if node.depth == self.max_depth:
                # Fixme: custom design, need to be more general
                assert node.value is None, 'Error: node value={}'.format(node.value)
                if node.is_terminal:
                    if node.state != 15:
                        node.value = -1
                    else:
                        node.value = 1
                node.value = self.heuristic_value(node, deepcopy(env))
            elif node.is_terminal:
                assert node.value is None, 'Error: node value={}'.format(node.value)
                #Fixme: custom design, need to be more general
                #instant_reward = env.P[node.parent.parent.state][node.parent.action][0][2]
                if node.state != 15:
                    instant_reward = -1
                else:
                    instant_reward = 1
                #print(env.P[node.parent.parent.state][node.parent.action], node.parent.parent.state, node.parent.action, node.state)
                #print("check instance reward:",node.parent.parent.state, node.parent.action, env.P[node.parent.parent.state][node.parent.action])
                #if instant_reward == 0 and env.P[node.parent.parent.state][node.parent.action][0][3]:
                #    instant_reward = -1
                node.value = instant_reward
                #print("check instance reward2:",node.state, node.value, env.P[node.parent.parent.state][node.parent.action])
                #node.value = env.instant_reward(node.parent.parent.state, self.t_call, node.parent.action, node.state)
            else:
                v = -1e99
                for ch in node.children:
                    v = max(v, self.minimax(ch, env))  # max operator
                assert node.value is None, 'Error: node value={}'.format(node.value)
                node.value = v
        else:  # ChanceNode
            self.set_worst_case_distribution(node, env)  # min operator
            v = 0.0
            #print("!!!!!!!!!!!!!!!!!!!!!")
            #print(node.parent.state, node.action)
            for ch in node.children:
            #    print(ch.state, ch.weight, ch.value)
                v += ch.weight * ch.value
            #print("#######################")
            v *= self.gamma
            # pessimistic expected reward value

            #FIXME : this part is custom made, needs to be checked when env is changed
            reachable_states = []
            R = 0.0
            for index, a in enumerate([node.action, (node.action + 1) % 4, (node.action - 1) % 4]):
                r_i = env.P[node.parent.state][a][0][2]
                if r_i == 0 and env.P[node.parent.state][a][0][3]:
                    r_i = -1
                R += r_i * env.transition_prob[index]
            v += R - L_r * tau * node.depth
            #v += R
            #v += env.expected_reward(node.parent.state, node.action) - L_r * tau * node.depth
            assert node.value is None, 'Error: node value={}'.format(node.value)
            node.value = v
        return node.value

    def set_worst_case_distribution(self, node, env):
        """
        Modify the weights of the children so that the worst case distribution is set wrt their values.
        :param node: ()
        :param env: () environment
        :return: None
        """
        assert type(node) is ChanceNode, 'Error: node type={}'.format(type(node))
        # Set values recursively
        for ch in node.children:
            self.minimax(ch, env)
        # Set weights
        L_p = 1
        tau = 1
        c = node.depth * L_p * tau
        # indexes = np.asarray(list(ch.state.index for ch in node.children))
        v = np.asarray(list(ch.value for ch in node.children))
        w0 = np.asarray(list(ch.weight for ch in node.children))
        #d = env.distances_matrix(np.asarray(list(ch.state for ch in node.children)))
        #w = distribution.worstcase_distribution_direct_method(v, w0, c, d)
        w = self.worstcase_distribution_direct_method(v, w0, c)
        for i in range(len(w)):
            node.children[i].weight = w[i]

    def worstcase_distribution_direct_method(self, v, w0, c):

        # def close(a, b, r=13):
        #     return isclose(round(a, r), round(b, r), rel_tol=1e-12, abs_tol=0.0)
        #
        # def closevec(u, v, r=13):
        #     assert len(u) == len(v), 'Error: vectors have different lengths: len(u)={} len(v)={}'.format(len(u), len(v))
        #     for i in range(len(u)):
        #         if not close(u[i], v[i], r):
        #             return False
        #     return True
        #
        # n = len(v)
        # if close(c, 0.0) or closevec(v, v[0] * np.ones(n)):
        #     return w0
        n = len(v)
        w_worst = np.zeros(n)
        w_worst[np.argmin(v)] = 1.0
        #if (wass_dual(w_worst, w0, d) <= c):
        return w_worst
        #lbd = c / wass_dual(w0, w_worst, d)
        #w = w_an = (1.0 - lbd) * w0 + lbd * w_worst
        #return clean_distribution(w)



    def heuristic_value(self, node, env):
        """
        Return the heuristic value of the input node.
        -- Ongoing work --
        :param node: ()
        :param env: () environment
        :return:
        """
        '''
        assert type(node) == DecisionNode, 'Error: node type={}'.format(type(node))
        value = 0.0
        s = node.state
        for t in range(self.horizon):
            a = self.action_space.sample()
            s, r, done = env.transition(s, a, is_model_dynamic=False)
            value += self.gamma**t * r
        return value - self.L_v * env.tau * node.depth
        '''
        return 0.0

    def get_depth_list(self, node, d_list):
        """
        Build the list containing the depth of each node in the tree.
        :param node: ()
        :param d_list: ()
        :return: None
        """
        d_list.append(node.depth)
        for ch in node.children:
            self.get_depth_list(ch, d_list)

    def act(self, env, done):
        """
        Compute the entire RATS procedure
        :param env: () environment
        :param done: (bool) terminal state or not
        :return: computed action
        """
        #self.t_call = env.get_time()
        self.t_call = 0
        #env = deepcopy(env)
        root = self.initialize_tree(deepcopy(env), done)
        self.minimax(root, deepcopy(env))
        return max(root.children, key=node_value).action
