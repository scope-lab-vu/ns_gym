import gymnasium as gym
from ns_gym.wrappers import *
from ns_gym.schedulers import *
from ns_gym.update_functions import *
from ns_gym.utils import type_mismatch_checker
import numpy as np
from ns_gym.benchmark_algorithms import MCTS
import time


# Create environment
env = gym.make("CartPole-v1",max_episode_steps=1000)

# Apply NS-Gym wrapper
scheduler = ContinuousScheduler(start=100,end=100)
#U_fn_2 = NoUpdate(scheduler)

U_fn_2 = IncrementUpdate(scheduler,1000)
tunable_params = {"gravity": U_fn_2}
ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=False)

ns_env.unwrapped.theta_threshold_radians = 23 * math.pi * 2 / 360

# Wrap environment with video recorder

reward_list = []
num_episodes = 1

trace = []

for i in range(num_episodes):
    episode_reward = 0
    done = False
    truncated = False
    obs, _ = ns_env.reset()
    obs,_ = type_mismatch_checker(obs,reward=None)
    max_steps = 1000
    step = 0

    print("Starting CartPole Execution.")

    while True:
        planning_env = ns_env.get_planning_env()
        agent = MCTS(planning_env, obs, d=100, m=100, c=1.44, gamma=0.99)
        action, _ = agent.search()
        
        next_obs, reward, done, truncated, info = ns_env.step(action)
        if done or truncated or step > max_steps:
            break
        next_obs, reward = type_mismatch_checker(next_obs, reward)
        trace.append((obs,action,reward,next_obs))

        obs = next_obs
        episode_reward += reward
        step += 1

    print(f"Episode {i} reward: {episode_reward}")
    reward_list.append(episode_reward)

print(f"Success Rate for MCTS Agent on Custom Env: {np.mean(reward_list)}")


# Create environment
env = gym.make("CartPole-v1",render_mode="human",max_episode_steps=1000)

# Apply NS-Gym wrapper
scheduler = ContinuousScheduler()

scheduler = ContinuousScheduler(start=100,end=100)
#U_fn_2 = NoUpdate(scheduler)
U_fn_2 = DecrementUpdate(scheduler,4)
tunable_params = {"gravity": U_fn_2}
ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=False)


# Wrap environment with video recorder

episode_reward = 0
done = False
truncated = False
obs, _ = ns_env.reset()

ns_env.unwrapped.state = trace[0][0]

ns_env.unwrapped.theta_threshold_radians = 23 * math.pi * 2 / 360
obs,_ = type_mismatch_checker(obs,reward=None)
max_steps = 100
step = 0
count = 1
for i in range(0,len(trace)):
    action = trace[i][1]
    
    next_obs, reward, done, truncated, info = ns_env.step(action)
    ns_env.render()

    if count == 1:
        time.sleep(5)
        count = 100

    step += 1
    if done:
        print("steps ", step)
        break

  



