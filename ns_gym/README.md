# NS-Gym: Open-Source Simulation Environments and Benchmarks for Non-Stationary Markov Decision Processes

NS-Gym is a set of wrappers for the popular gymnasium environment to model non-stationary Markov decision processes.

# Install from source

Requires `python = 3.10`. To install on MacOS and Linux make sure you are in the same directory as this README file then execute the following lines in the terminal.

```bash
python3.10 -m venv env
source env/bin/activate
pip install .
```

# Quickstart
Suppose we want to model a non-stationary environment in the classical CartPole environment, where the pole’s mass increases by 0.1 units at each time step, and the system’s gravity increases through a random walk every three time steps. Furthermore, we want the decision-making agent to have a basic notification level. The following code snippet shows the general experimental setup in this CartPole Gymnasium environment using NS-Gym.

```python

###### Step 1: Import necessary gym and ns_gym modules
import gymnasium as gym
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate


###### Step 2: Create a standard gym environment ####
env = gym.make("CartPole-v1")
#############

########## Step 3: to describe the evolution of the non-stationary parameters, 
# we define the two schedulers and update functions that model the semi-Markov chain over the relevant parameters
############
scheduler_1 = ContinuousScheduler()
scheduler_2 = PeriodicScheduler(period=3)

update_function1= IncrementUpdate(scheduler_1, k=0.1)
update_function2 = RandomWalk(scheduler_2)

##### Step 4: map parameters to update functions
tunable_params = {"masspole":update_function1, "gravity": update_function2}


######## Step 5: set notification level and pass environment and parameters into wrapper
ns_env = NSClassicControlWrapper(env,tunable_params,change_notification=True)



######### Step 6: set up ns-environment and agent interaction loop. i.e ... 
done = False
truncated = False

episode_reward = 0

obs,info = ns_env.reset()

while not done and not truncated:
    planning_env = ns_env.get_planning_env()
    mcts_agent = ns_gym.benchmark_algorithms.MCTS(env=planning_env, state=obs.state, gamma=1, d=500, m=100, c=2)
    action, action_vals = mcts_agent.search()
    obs,reward,done,truncated,info = ns_env.step(action)
    episode_reward += reward.reward


print("Episode Reward: ", episode_reward)
```


# Tutorial:

A more comprehensive tutorial can be found [here](tutorial.ipynb)


