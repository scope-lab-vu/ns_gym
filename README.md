# NS-Gym: A Comprehensive and Open-Source Simulation Framework for Non-Stationary Markov Decision Processes

**NS-Gym** (Non-Stationary Gym) is a flexible framework providing a standardized abstraction for both modeling Non-Stationary Markov Decision Processes (NS-MDPs) and the key problem types that a decision-making entity may encounter in such environments.

Built on top of the popular Gymnasium library, NS-Gym provides a set of wrappers for existing environments, making it easy to incorporate non-stationary dynamics and manage the nature of agent-environment interaction specific to NS-MDPs.

A key feature of NS-Gym is emulating the core problem types of decision-making in non-stationary settings; these problem types concern not only the ability to adapt to changes in the environment but also the ability to detect and characterize these changes. 

We currently support the Gymnasium classic control suite of environments, MuJoCo environments, and stochastic environments like FrozenLake. NS-Gym is designed to be easily extensible, allowing users to create their own non-stationary environments by defining parameter change schedules and update functions. We welcome contributions from the community to expand the library of non-stationary environments and algorithms!

---

## Project Webpage and Documentation

Visit our project webpage for tutorials, core concepts, and documentation: [nsgym.io](https://nsgym.io) (to be live soon)

---

## Installation

To install NS-Gym, you can use `pip` directly from GitHub. uv is recommended but not required:

```bash
uv pip install git+https://github.com/scope-lab-vu/ns_gym
```

We'll eventually release NS-Gym on PyPI for easier installation.

-----

## White Paper

**NS-Gym: Open-Source Simulation Environments and Benchmarks for Non-Stationary Markov Decision Processes**

[Read the the preprint on ArXiv](https://arxiv.org/abs/2501.09646) (An updated version to be published at NeurIPS 2025 Dataset and Benchmarks track is coming soon).

### Citation

```bibtex
@article{keplinger2025ns,
  title={NS-Gym: Open-Source Simulation Environments and Benchmarks for Non-Stationary Markov Decision Processes},
  author={Keplinger, Nathaniel S and Luo, Baiting and Bektas, Iliyas and Zhang, Yunuo and Wray, Kyle Hollins and Laszka, Aron and Dubey, Abhishek and Mukhopadhyay, Ayan},
  journal={arXiv preprint arXiv:2501.09646},
  year={2025}
}
```

-----

## Decision Making Algorithm Support

NS-Gym is designed to be compatible with existing reinforcement learning libraries such as [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/). Additionally, NS-Gym provides baseline algorithms designed explicitly for non-stationary environments, as well as a leaderboard to compare algorithm performance on various non-stationary tasks.

-----

##  NS-Gym in Action

Here are three examples of non-stationary environments created using NS-Gym. Each demonstrates a transition from an initial MDP ($\mathcal{MDP}_0$) to a modified MDP ($\mathcal{MDP}_1$) by changing environment parameters ($\theta_0 \rightsquigarrow \theta_1$).

| Stationary MDP | Change Type | Non-Stationary MDP |
| :---: | :---: | :---: |
| **CartPole:** Stationary MDP | **At timestep $t$ gravity massively increases according to a user-defined step function.** | **CartPole:** Non-Stationary MDP |
| ![CartPole Stationary](docs/source/_static/images/cartpole_stationary.gif) | $\Large \theta_0 \rightsquigarrow \theta_1$ | ![CartPole Non-Stationary](docs/source/_static/images/cartpole_nonstationary.gif) |
| **FrozenLake:** Stationary MDP | **Probability of moving in the intended direction goes to 0 just before reaching the goal.** | **FrozenLake:** Non-Stationary MDP |
| ![FrozenLake Stationary](docs/source/_static/images/frozen_lake_stationary.gif) | $\Large \theta_0 \rightsquigarrow \theta_1$ | ![FrozenLake Non-Stationary](docs/source/_static/images/frozenlake_nonstationary.gif) |
| **Ant:** Stationary MDP | **Magnitude of gravity gradually decreases at each timestep following a geometric progression.** | **Ant:** Non-Stationary MDP |
| ![Ant Stationary](docs/source/_static/images/ant_stationary.gif) | $\Large \theta_0 \rightsquigarrow \theta_1$ | ![Ant Non-Stationary](docs/source/_static/images/ant_non_stationary.gif) |

*Note: This type of parameter shift is just one example of how an NS-MDP can be implemented. The policies controlling the agents are detailed in the full documentation.*

------


# Quickstart
Suppose we want to model a non-stationary environment in the classical CartPole environment, where the pole’s mass increases by 0.1 units at each time step, and the system’s gravity increases through a random walk every three time steps. Furthermore, we want the decision-making agent to have a basic notification level. The following code snippet shows the general experimental setup in this CartPole Gymnasium environment using NS-Gym.

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

# Tutorial:

A more comprehensive tutorial can be found [here](tutorial.ipynb)

# Development and testing

We welcome any contributions to this NS-Gym project! If you find a bug or want to add a new feature, please feel free to open an issue or submit a pull request.

Clone the repository, install the required dependencies in editable mode, and run the tests to ensure everything is working correctly. We use UV for package management. 

```bash
git clone https://github.com/scope-lab-vu/ns_gym.git
cd ns_gym
uv pip install -e ".[all]" --force-reinstall
```

To run all test in the project run: 

```bash
pytest tests/
```




