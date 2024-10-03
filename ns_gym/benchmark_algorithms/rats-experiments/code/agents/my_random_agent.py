"""
A Random Agent given as an example
"""

from gym import spaces
from ..utils.utils import assert_types


class MyRandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p is not None:
            assert_types(p, [spaces.discrete.Discrete])
            self.__init__(p[0])

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying Random agent:')
        print('Action space       :', self.action_space)

    def act(self, observation=None, reward=None, done=None):
        return self.action_space.sample()
