import gymnasium as gym
import ns_gym as nsg
import numpy as np
import copy
from gymnasium import spaces
from typing import List, Optional



class NS_Routing(gym.Env):
    """Routing evnironment in a real city network. 

    The weights are the travel time between each pair of nodes. Non-stationarity is introduced by changing the weights of the edges. 
    The agent has to find the shortest path between the source and the destination as the weights change.

    We load a graph form OSM and use the weights as the travel time between each pair of nodes.


    Observation space:
        Type: Discrete
        Size: nS

    Actions:
        Type: Discrete
        Num: nA
    """
    def __init__(self):
        super().__init__()

    def step(self, action):

        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError
    
    



