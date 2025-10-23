import numpy as np
import torch
import gymnasium as gym
from ns_gym import wrappers, benchmark_algorithms, update_functions, schedulers, base
import logging


class Evaluator:
    """
    Evaluator class for evaluating the performance of the model. Should take in model agent and environment and return the performance of the model
    """
    def __init__(self, agent: benchmark_algorithms.Agent, env: gym.Env, episodes: int = 1000):
        pass
    
    def evaluate(self) -> float:
        pass



