# Quickstart Guide

## Installation

Create a virtual environment (optional but recommended). Here we use `uv` for development, but you can also use `venv` or `conda`.

```bash
uv venv
source .venv/bin/activate 
```

To install NS-Gym, you can use pip (remove the `uv` prefix if not using UV):

```bash
uv pip install ns-gym
```

for nightly builds, you can install directly from the GitHub repository:

```bash
uv pip install git+https://github.com/scope-lab-vu/ns_gym
```

## Building a non-stationary environment

We can build a non-stationary environment in six lines of code. First, we import the necessary modules from NS-Gym and Gymnasium:

```python
import ns_gym
import gymnasium as gym
```

We will then import the necessary wrappers to create a non-stationary environment. As you would in Gymnasium, we create an environment.

```python
env = gym.make("CartPole-v1")
```

We then must define how evironmental parameters will evolve over time to induce non-stationarity. To do this NS-Gym provides a suite of "schedulers" and "update functions". Schedulers define *when* a parameter is updated, while update functions define *how* a parameter is updated. In this example, we will use a `ContinuousScheduler` to update the pole's mass at every time step, and a `PeriodicScheduler` to update the gravity every three time steps. We will use an `IncrementUpdate` function to increase the pole's mass by 0.1 units at each update, and a `RandomWalk` function to change the gravity.

We define the schedulers and update functions and map them to the environmental parameter names. See environments documentation for complete table of environmental parameter than can be tuned.


```python
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate

########## Describe the evolution of the non-stationary parameters, 
# we define the two schedulers and update functions that model the semi-Markov chain over the relevant parameters
############
scheduler_1 = ContinuousScheduler()
scheduler_2 = PeriodicScheduler(period=3)

update_function1= IncrementUpdate(scheduler_1, k=0.1)
update_function2 = RandomWalk(scheduler_2)

#Map parameters to update functions
tunable_params = {"masspole":update_function1, "gravity": update_function2}

```
We then pass the mapped parameters and the base environment into the `NSClassicControlWrapper`, which creates the non-stationary environment. We also have to choose whether we want the decision making agent to be able to receive notifications about changes in the environment. The `change_notification` flag is set to `True`, so that the agent is notified when a change in the environment occurs but not the specific details of the change. If we wanted to provide more detailed information about the changes, we could set the `delta_change_notification` flag to `True` as well which would provide the agent with the magnitude of the change for each parameter.


```python
from ns_gym.wrappers import NSClassicControlWrapper
# Pass environment and parameters into wrapper
ns_env = NSClassicControlWrapper(env,tunable_params,change_notification=True)
```

We can then evalute our decision-making policies in this non-stationary environment. Here, we use NS-Gym's Monte Carlo Tree Search (MCTS) implementation as an example. 

```python
from ns_gym.benchmark_algorithms import MCTS

done = False
truncated = False
episode_reward = 0

obs,info = ns_env.reset()

# We get the planning environment for the MCTS agent -- this environment does not have non-stationarity enabled and it a static snapshot of the current state environment
planning_env = ns_env.get_planning_env()

mcst_agent = MCTS(planning_env, state=obs["state"], d=50, m=100,c=1.4,gamma=0.99)
done = False
truncated = False

timestep = 0
while not (done or truncated):
    action = mcst_agent.act(obs,planning_env)
    obs, reward, done, truncated, info = ns_env.step(action)


    if timestep % 10 == 0:
        print("Timestep: ", timestep)
        print("obs: ", obs)
        print("reward: ", reward)   
        print("########")
        print("\n")
    planning_env = ns_env.get_planning_env()
    episode_reward += reward.reward
    timestep += 1

print("Episode Reward: ", episode_reward)

```

The environment observation and reward at timestep 0 may look like this:
```python
Timestep:  0

obs:  {'state': array([-0.03006991,  0.19717823,  0.02711801, -0.3215324 ], dtype=float32), 
        'env_change': {'masspole': 1, 'gravity': 1}, 
        'delta_change': {'masspole': 0.0, 'gravity': 0.0}, 
        'relative_time': 1}

reward:  Reward(reward=1.0, 
                env_change={'masspole': 1, 'gravity': 1}, 
                delta_change={'masspole': 0.0, 'gravity': 0.0}, 
                relative_time=1)
```     

The `obs` dictionary of the following terms:

- `state`: the standard Gymnasium observation of the environment.
- `env_change`: a dictionary indicating whether this parameter has changed (1 indicates a change, 0 indicates no change). This is only available if `change_notification=True` is set in the wrapper.
- `delta_change`: a dictionary indicating the magnitude of change for each parameter. This is only available if `delta_change_notification=True` is set in the wrapper.
- `relative_time`: Current time step of environment.

The `reward` object is a data class rather than a dictionary that contains the same terms. While the observation is a dictionary to maintain compatibility with Gymnasium, the reward is a dataclass to allow for easier extension in the future for non-stationary rewards while working with a more robust data structure.


---
## Complete Code Example:

```python
###### Step 1: Import necessary gym and ns_gym modules
import gymnasium as gym
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from ns_gym.benchmark_algorithms import MCTS


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

planning_env = ns_env.get_planning_env()
mcst_agent = MCTS(planning_env, state=obs["state"], d=50, m=100,c=1.4,gamma=0.99)
done = False
truncated = False

timestep = 0
while not (done or truncated):
    action = mcst_agent.act(obs,planning_env)
    obs, reward, done, truncated, info = ns_env.step(action)


    if timestep % 10 == 0:
        print("Timestep: ", timestep)
        print("obs: ", obs)
        print("reward: ", reward)   
        print("########")
        print("\n")
    planning_env = ns_env.get_planning_env()
    episode_reward += reward.reward
    timestep += 1

print("Episode Reward: ", episode_reward)
```