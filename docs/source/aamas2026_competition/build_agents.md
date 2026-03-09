# Build Your Agent


Welcome! This is the second tutorial in a series for the full submission workflow:

1.  [Environment Setup](setup.md): create your repository, configure Python, and build Docker images.
2.  This tutorial: implement a model-based or model-free agent and register it.
3. [Create a Custom Environment](make_env.md): define non-stationarity with schedulers and update functions.
4. [Submit Your Agent](submission.md): run final checks and send your repository for evaluation.

This is the part you will spend most time on. Let's get started!

> “The gem cannot be polished without friction, nor man perfected without trials.” – Lucius Annaeus Seneca


## Overview
In general, Ns-Gym do not limit the competitors on what type of agents to implement, including but not limited to reinforcement learning, online planning, meta-learning, and continuous learning. We also provides two types of agents to implement:

- **Model-based** agents implement `get_action(obs, planning_env)`.
- **Model-free** agents implement `get_action(obs)`.


The competition template provides three base environments for evaluation: [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/), [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/), and [Ant-v5](https://gymnasium.farama.org/environments/mujoco/ant/). Some environments might achieve better result for model-based approaches, some better in model-free approaches. We will take them into account in the leaderboard. Also, all environments are able to create customizable non-stationarity settings. For more, see [Create a Custom Environment](make_env.md).



## 1. Implement a model-based agent

First, we take a look at `agent.py`. There are two methods in `MyModelBasedAgent`:

- `get_action(obs, planning_env)` (**required**)
    - `obs` contains the current observation.
    - `planning_env` is a stationary snapshot provided by NS-Gym.

- `set_seed(seed)` (optional, but useful for testing)

<details>
  <summary>What is planning_env?</summary>
  
  NS-Gym provides an interface to grab a **stationary** version of the environment for planning by calling `ns_env.get_planning_env()`. This returns a copy of the environment in accordance with the notification settings. If `delta_change_notification` is set to `True`, the most up-to-date version of the environment is returned (though no future evolutions). Otherwise the first known version of the MDP is returned.

</details>

\
For this tutorial, we will use [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) as the model-based algorithm. In `agent.py`, we borrow code from our benchmark and `mcts_example.py` to fill in `MyModelBasedAgent`. It should look like:

```python
from typing import Dict

from ns_gym.benchmark_algorithms import MCTS
from AAMAS_Comp.base_agent import ModelBasedAgent


class MyModelBasedAgent(ModelBasedAgent):
    def __init__(self, d=50, m=100, c=1.4, gamma=0.99) -> None:
        super().__init__()
        self.d = d
        self.m = m
        self.c = c
        self.gamma = gamma

    def get_action(self, obs: Dict, planning_env):
        state = obs["state"]
        solver = MCTS(
            env=planning_env,
            state=state,
            d=self.d,
            m=self.m,
            c=self.c,
            gamma=self.gamma,
        )
        action, _ = solver.search()
        return action
```

## 2. Implement a model-free agent

Model-free agents do not receive `planning_env`. Instead, they act directly from observations and learn from interaction.

```python
def get_action(self, obs):
    # Replace with your policy logic.
    return action
```
Similar to the model-based case, we use [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) to evluate. Please visit `ppo_example.py` and `example_submission.py` for example.

## 3. Register your agent in `submission.py`

After implementing either algorithms, we look inside `get_agent(env_id)` to return the correct agent for each environment. For example, if we want to apply MCTS to the `FrozenLake-v1` environment:

```python
if env_id == "FrozenLake-v1":
    return MyModelBasedAgent(d=50, m=100, c=1.4, gamma=0.99)
```

If you want the model to load offline weights, we encourage you load weights in `submission.py`. Refer to the `Ant-v5` environment in `example_submission.py` for details.

## 4. Running Your Algorithm

To run the Run the evaluator locally:

```bash
uv run python evaluator.py
```

The evaluator supports the following flags:
- `--num-episodes`: how many episodes to evaluate
- `--start-seed`: what is the starting seed. Note you could also change this by using `set_seed(seed)` in `agent.py`.

To run the program using Docker, 

```bash
docker compose run --rm test-submission
```

Using either way, the results should be populated in the `results/` folder.

## Outcome

In this tutorial, you have:

- built a model-based agent
- built a model-free agent
- ran the code both locally and using Docker.

Need help? Join in our [Office Hour sessions](../aamas2026_competition.md#office-hours)!

In the next tutorial, we will cover how to create a custom, non-stationary environment. See you there!
