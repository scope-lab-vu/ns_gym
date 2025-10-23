import gymnasium as gym
from typing import Any, Type, Union
from ns_gym import benchmark_algorithms,wrappers,schedulers,update_functions,base


"""Not sure how to sturcture all the benchmaking yet...

Replicating the MCTS experiments from the Luo et al. AMAS 2024 paper. 

In Yunuo's paper the MCTS runs for 25-3000 iterations on frozen lake. 
"""

def make_env(env : gym.Env, 
             wrapper : base.NSWrapper,
             scheduler : schedulers.Scheduler, 
             update_fn : update_functions.UpdateFunction,
             tunable_params : dict[str, Any],
             max_iters : int = 3000) -> gym.Env:
    pass

def main():
    pass