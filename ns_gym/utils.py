# import ns_bench.base as base
import math
import numpy as np
from scipy.stats import wasserstein_distance
from copy import deepcopy
import ns_gym.base as base

def update_probability_table(P,
                             nS: int,
                             nA: int, 
                             T:int,
                             transition_fn):
    """Update the fulle probability table to include time axis

    Args:
        P (base.stationary_probablity_table): _description_
        nS (int): _description_
        nA (_type_): _description_
        int (_type_): _description_
        T (int): _description_
        transition_fn (base.NSTransitionFn): _description_

    Returns:
        _type_: _description_
    """
    
    time_augmented_table = {t:{s:{a : [] for a in range(nA)} for s in range(nS)} for t in range(T)}

    for t in range(T):
        pt = deepcopy(P)
        for s in range(nS):
            for a in range(nA):
                transitions = pt[s][a]
                updated_probs, change= transition_fn(transitions,t)
                for i,tup in enumerate(transitions):
                    temp = list(tup)
                    temp[0] = updated_probs[i]  
                    temp.append(change)
                    transitions[i] = tuple(temp)
                time_augmented_table[t][s][a] = transitions

    return time_augmented_table                                                   


def state_action_update(transitions: list,new_probs: list):
    """Update the transitions associated with a single state action pair in the gridworld environments with new probs
    Notes:
        The possible transitions from state s with action a are typically stored in a list of tuples. 
        For example, the possible transitions for (s,a) in FrozenLake are store in table P at P[s][a], 
        where:
        
        P[s][a] = [(p0, newstate0, reward0, terminated),
                        (p1, newstate1, reward1, terminated),
                        (p2, newstate2, reward2, terminated)]

        The indended direction is stored in P[s][a][1].
    """

    for i,tup in enumerate(transitions):
        temp = list(tup)
        temp[0] = new_probs[i]
        transitions[i] = tuple(temp)
    return transitions



def n_choose_k(n, k):

    """Calculate the binomial coefficient 'n choose k'
    Args:
        n (int): Total number of items.
        k (int): Number of items to choose.
    
    Returns:
        int: The binomial coefficient, i.e., the number of ways to choose k items
            from a set of n items.
    """

    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def wasserstein_dual(u:np.ndarray,v:np.ndarray):
    """_summary_

    Args:
        u (np.ndarray): _description_
        v (np.ndarray): _description_
        d (np.ndarray): _description_

    Returns:
        _type_: _description_
    """

    dist = wasserstein_distance(u,v)

def categorical_sample(probs:list):
    """Sample from a categorical distribution
        Args: 
            probs (list): A list of probabilities that sum to 1
        Returns:
            int: The index of the sampled probability
    """
    return np.random.choice(len(probs),p=probs)


# def update_frozen_lake_table(P,nS,nA):
#     """Update the transition table for the FrozenLake environment
#     Args:
#         P : probability table for the FrozenLake environment. P[s][a] = [(p0, newstate0, reward0, terminated),
#                         (p1, newstate1, reward1, terminated),
#                         (p2, newstate2, reward2, terminated)
#     Returns:
#         _type_: _description_
#     """
#     for s in range(nS):
#         for a in range(nA):
#             transitions = P[s][a]
#             for i


def type_mismatch_checker(observation=None, reward=None):
    """
    A helper function to handle type mismatches between ns_gym
    and Gymnasium environments.

    Args:
        observation: The observation object, which may be an instance of nsg.base.Observation.
        reward: The reward object, which may be an instance of nsg.base.Reward.

    Returns:
        A tuple containing the processed observation and reward, with None if not provided.
    """
    # Process the observation if provided
    if observation is not None:
        if isinstance(observation, base.Observation):
            obs = observation.state
        else:
            obs = observation
    else:
        obs = None

    # Process the reward if provided
    if reward is not None:
        if isinstance(reward, base.Reward):
            rew = reward.reward
        else:
            rew = reward
    else:
        rew = None

    return obs, rew


if __name__ == "__main__":
    test  = n_choose_k(5,2)
