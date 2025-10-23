import gymnasium as gym
import numpy as np
import random
from typing import Type, Any, List
from ns_gym import benchmark_algorithms,wrappers,update_functions,schedulers,utils

"""Test vanilla MCTS on Inter-Episodic non-stationary environments. I follow the experimental setup from Pettet and Zhang et al. (2024)
"""
    
def make_env(dist):
    env = gym.make("FrozenLake-v1",is_slippery=False)
    fl_scheduler = schedulers.ContinuousScheduler()
    fl_update_fn = update_functions.DistributionNoUpdate(fl_scheduler)
    param = {"P":fl_update_fn}
    env = wrappers.NSFrozenLakeWrapper(env,param,change_notification=True,delta_change_notification=True,in_sim_change=False,initial_prob_dist=dist)    
    return env


def run_experiment(env, mcts_agent:Type[benchmark_algorithms.MCTS], seeds : List[int], iterations, tree_gamma=0.99,exploration_constant=np.sqrt(2)):
    """Run a single episode of the MCTS agent on the environment. 

    Args:
        env (gym.Env): The environment to run the agent on
        mcts_agent (Type[benchmark_algorithms.MCTS]): The MCTS agent to run
        seed (int): The seed to use for the environment
        iterations (int): The number of iterations to run the MCTS agent for
        tree_gamma (float, optional): The discount factor for the MCTS tree. Defaults to 0.99.
        exploration_constant (float, optional): The exploration constant for the MCTS tree. Defaults to np.sqrt(2).
    
    Returns:
        float: The cummulative reward of the agent
    """

    total_reward = [] 
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
        cummulative_reward = 0
        done = False
        while not done:
            agent = mcts_agent(obs.state, obs.relative_time, env, tree_gamma=tree_gamma, exploration_constant=exploration_constant)
            agent.search(iterations)
            action = agent.best_action()
            obs, reward, done, _, info = env.step(action)
            cummulative_reward += reward.reward
        total_reward.append(cummulative_reward)
    return np.mean(total_reward), np.std(total_reward)


def main(dists,seeds,iterations = 100, tree_gamma=0.99,exploration_constant=np.sqrt(2)):
    for d in dists:
        env = make_env(d)
        mcts_agent = benchmark_algorithms.MCTS
        mean_reward, std_reward = run_experiment(env, mcts_agent, seeds, iterations, tree_gamma, exploration_constant)
        env.close()
        print(f"Mean reward: {mean_reward}, std reward: {std_reward}")






if __name__ == "__main__":

    intended_prob = [1/3]
    dists = [[p,(1-p)/2,(1-p)/2] for p in intended_prob]
    main(dists,
         seeds = list(range(1)),
         iterations = 100,
         tree_gamma=0.99,
         exploration_constant=np.sqrt(2))






