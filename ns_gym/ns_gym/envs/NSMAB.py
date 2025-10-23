import gymnasium as gym
import numpy as np 
import ns_gym.base as base

"""
This is a simple MAB environment where the rewards are generated from a normal distribution
"""

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

class MAB(gym.Env):
    """A MAB gridworld environment. 

    A agent begins in the center of a 5 x 5 grid. The agent can move up, down, left, or right.
    The agent has four goal locations it can visit at the corners of the grid to to receive a reward. 
    Each reward is drawn from a normal distribution with some mean and variance. 

    ## State Space

    The agent 25 possible locations in the grid. 

    ## Action Space

    We have a simple action space with four actions: up, down, left, right.

    ## Reward

    The agent receives a reward when it reaches one of the four goal locations. 
    The agent must then leave the goal location and return to the center of the grid to receive another reward.

    ## Termination

    The environment termnates after a fixed number of time steps.

    Args:
        gym (_type_): _description_
    """
    def __init__(self) -> None:
        super().__init__()

        self.nS = 25
        self.nA = 4
        self.ncol = 5
        self.nrow = 5 

        self.action_space = gym.spaces.Discrete(self.nA)
        self.observation_space = gym.spaces.Discrete(self.nS)
        
        self.state = self.encode((2,2)) #The starting state is the center of the grid.

    def inc(self,action: int):
        """Given an action, return the new state after taking the action.

        Args:
            action (int): Action to take.
        """
        x,y = self.decode(self.state)
        if action == LEFT:
            x = max(0,x-1)
        elif action == DOWN:
            y = min(self.nrow-1,y+1)
        elif action == RIGHT:
            x = min(self.ncol-1,x+1)
        elif action == UP:
            y = max(0,y-1)
        return self.encode((x,y))

    def step(self,action: int):
        self.state = self.inc(action)

    def encode(self,coords: tuple[int,int]) -> int:
        """Encode the grid coordinates into a single integer.
        """
        x,y = coords
        return x + y * self.ncol        


    def decode(self,s:int) -> tuple[int,int]:
        """Decode the single integer into grid coordinates.
        """
        x = s % self.ncol
        y = s // self.ncol
        return x,y

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def unwrapped(self):
        pass

