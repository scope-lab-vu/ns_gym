import gymnasium as gym
import ns_gym as nsg
import numpy as np
import copy
from gymnasium import spaces
from typing import Optional


"""
"""

MAPS = {"bridge": ["HHHHHHHH", "FFFFFHHH", "GFFFSFFG", "FFFFFHHH", "HHHHHHHH"]}

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class Bridge(gym.Env):
    """Bridge environment
    This is a simple implementation of the bridge environment used in Lecarpentier and Rachelson
    (2019). The primary difference between this implementation and the one used in the paper is that the non-sationarity is handled externally by the ns
    The agent starts at the position marked by "S" and has to reach the goal position marked by "G". The agent can move in four directions: up, down, left, right. The agent receives a reward of -1 for stepping on a hole "H" and a reward of 1 for reaching the goal "G". The episode ends when the agent reaches the goal or steps on a ho

    Args:
        gym (_type_): _description_
    """

    def __init__(
        self,
        global_init_probs=[1, 0, 0],
        map_name="bridge",
        epsilon=0.5,
        split_probs=False,
    ):
        # super(Bridge, self).__init__()
        self.map = np.asarray(MAPS[map_name], dtype="c")
        self.global_init_probs = copy.copy(global_init_probs)
        self.P = global_init_probs
        self.epsilon = epsilon

        self.nS = self.map.size
        self.nA = 4
        self.nrow = len(self.map)
        self.ncol = len(self.map[0])

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.left_side_prob = copy.copy(global_init_probs)
        self.right_side_prob = copy.copy(global_init_probs)
        self.split_probs = split_probs
        self.delta = {
            LEFT: np.array([0, -1]),
            DOWN: np.array([1, 0]),
            RIGHT: np.array([0, 1]),
            UP: np.array([-1, 0]),
        }

    def step(self, action):
        if self.split_probs:
            P = self.get_loc_based_prob(self.state_to_coord(self.s))
        else:
            P = self.P

        action = np.random.choice(
            [action, (action + 1) % 4, (action - 1) % 4], p=self.P
        )
        newstate, reward, done = self.transition(action)
        self.s = newstate

        return int(self.s), int(reward), done, False, {"prob": P}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = self.coord_to_state(2, 4)
        return int(self.s), {"prob": self.P[0]}

    def transition(self, a):
        coord = self.state_to_coord(self.s)
        LEFT = 0
        DOWN = 1
        RIGHT = 2
        UP = 3

        delta = {
            LEFT: np.array([0, -1]),
            DOWN: np.array([1, 0]),
            RIGHT: np.array([0, 1]),
            UP: np.array([-1, 0]),
        }

        newcoord = coord + delta[a]

        if self.check_out_of_bound(newcoord[0], newcoord[1]):
            newcoord = coord

        newstate = self.coord_to_state(newcoord[0], newcoord[1])
        reward, done = self.get_reward(newcoord)
        return newstate, reward, done

    def check_out_of_bound(self, row, col):
        if row < 0 or row >= self.nrow:
            return True
        if col < 0 or col >= self.ncol:
            return True
        return False

    def coord_to_state(self, row, col):
        return row * self.ncol + col

    def state_to_coord(self, state):
        return np.array([state // self.ncol, state % self.ncol])

    def get_loc_basedProb(self, coord):
        if coord[1] < 4:
            return self.left_side_prob
        else:
            return self.right_side_prob

    def get_reward(self, coord):
        x, y = coord[0], coord[1]

        if self.map[x, y] == b"H":
            reward = -1
            done = True
        elif self.map[x, y] == b"G":
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        return reward, done

    def render(self):
        for i in range(self.nrow):
            for j in range(self.ncol):
                if (
                    self.state_to_coord(self.s)[0] == i
                    and self.state_to_coord(self.s)[1] == j
                ):
                    print("X", end=" ")
                else:
                    print(self.map[i, j].decode(), end=" ")
            print()
        print()

    @property
    def transition_matrix(self):
        return self._get_transition_matrix()

    @property
    def P(self):
        return self.transition_matrix()

    def _get_transition_matrix(self):
        table = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self.coord_to_state(row, col)
                for a in range(self.nA):
                    li = table[s][a]
                    letter = self.map[row, col]
                    if letter == b"H" or letter == b"G":
                        newstate = s
                        reward, done = self.get_reward([row, col])
                        li.append((1.0, newstate, reward, done))
                        continue
                    else:
                        for ind, b in enumerate([a, (a + 1) % 4, (a - 1) % 4]):
                            newcoord = np.array([row, col]) + self.delta[b]
                            if self.check_out_of_bound(newcoord[0], newcoord[1]):
                                newcoord = np.array([row, col])
                            newstate = self.coord_to_state(newcoord[0], newcoord[1])
                            reward, done = self.get_reward(newcoord)
                            li.append((self.P[ind], newstate, reward, done))
        return table


if __name__ == "__main__":
    import ns_gym as nsg
    import gymnasium as gym

    env = gym.make("ns_bench/Bridge-v0")
    # env = Bridge()
    obs, info = env.reset()
    env.render()
    done = False
    while not done:
        agent = nsg.benchmark_algorithms.MCTS(env, obs, d=100, m=100, c=1.44, gamma=0.9)
        a, _ = agent.search()
        obs, reward, done, truncated, info = env.step(a)
        env.render()
