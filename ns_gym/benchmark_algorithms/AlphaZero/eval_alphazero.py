import ns_gym as nsg
from ns_gym.benchmark_algorithms.AlphaZero import AlphaZeroNetwork
import numpy as np
import random 
import gymnasium as gym

model_checkpoint_path = "--"
alphazero_agent = nsg.benchmark_algorithms.AlphaZero.AlphaZeroAgent(action_space_dim=4,
                                                                    observation_space_dim=48,
                                                                    n_hidden_layers=3,
                                                                    n_hidden_units=96,
                                                                    model=AlphaZeroNetwork,
                                                                    lr=0.001,
                                                                    gamma=0.99,
                                                                    c=np.sqrt(2),
                                                                    num_mcts_simulations=200,
                                                                    max_mcts_search_depth=100,
                                                                    model_checkpoint_path=model_checkpoint_path)



######## Set up the environment ########
# env = gym.make("FrozenLake-v1",render_mode="ansi")
env = gym.make("CliffWalking-v0",render_mode="ansi")
# scheduler = nsg.schedulers.ContinuousScheduler()
# updateFn = nsg.update_functions.DistributionNoUpdate(scheduler=scheduler)
# param = {"P":updateFn}

# env = nsg.wrappers.NSFrozenLakeWrapper(env,param,change_notification=False,delta_change_notification=False,in_sim_change=False,initial_prob_dist=[1/3,1/3,1/3])

##### Evaluate the agent #####

seeds = random.sample(range(1000), 10)
total_rewards = []

for seed in seeds:
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    episdoe_reward = 0
    print(env.render())
    while not done and not truncated:
        action = alphazero_agent.act(obs,env)
        obs, reward, done, truncated, info = env.step(action)
        print(env.render())
        print(reward)
        episdoe_reward += reward
    total_rewards.append(episdoe_reward)


print("Average reward over 10 seeds: ", np.mean(total_rewards))
