# Create a Custom Non-Stationary Environment

Welcome! This is the third tutorial in a series for the full submission workflow:

1.  [Environment Setup](setup.md): create your repository, configure Python, and build Docker images.
2.  [Build Your Agent](build_agents.md): implement a model-based or model-free agent and register it.
3. This tutorial: define non-stationarity with schedulers and update functions.
4. [Submit Your Agent](submission.md): run final checks and send your repository for evaluation.

This is an in-depth tutorial on the settings of NS-Gym. Let's get started!

> “The beginning of wisdom is to call things by their proper name.” – Confucius

## Overview

A custom NS-Gym environment has three parts:

1. Base Gymnasium environment
2. Scheduler (roughly, `how often` changes happen)
3. Update function (roughly, `how much` change for each update)

We can also design the agent's "notification level", described in section 6.

## 1. Pick a base environment and wrapper
**Wrappers** introduces non-stationarity to Gym environments by sitting on top of each base environment. It applies **scheduled parameter updates** (schedulers and update functions) over time, and exposes the resulting environment through a consistent interface.
Use the wrapper that matches the environment family:

- `FrozenLake-v1` -> `NSFrozenLakeWrapper`
- `CartPole-v1` -> `NSClassicControlWrapper`
- `Ant-v5` -> `NSMujocoWrapper`

## 2. Define change timing with schedulers

**Schedulers** dictate *how often* the environment changes over time. Here are all scheduler choices avaliable in NS-Gym (for details, see `schedulers.py`):

- `ContinuousScheduler()`
- `PeriodicScheduler(period=n)`
- `DiscreteScheduler({t1, t2, ...})`
- `RandomScheduler(p)`
- `MemorylessScheduler(p)`
- `CustomScheduler(fn)`

## 3. Define parameter updates

**Update Functions** decide *how much* each update would be. The idea is we attach an update function to each tunable parameter. For details, see `single_param.py` and `distribution.py` for a single parameter change and transition function's distribution change, respectively.

There are many ways one could decide the update, including but not limited to:

- increment/decrement updates,
- random walk updates,
- bounded updates,
- distribution updates for stochastic environments.

## 4. Configure change notifications

Now, we have a environment where the non-stationarity is fully customizable. However, will the agent be notified of the change? We have designed the concept of **notification levels**:

- `change_notification=True`: the agent is notified of the change, but not the magnitudes
- `delta_change_notification=True`: both change and magnitude of the change is known by the agent

Confused by how the notification mechanism work? See this example below:

<details>
  <summary>Execution Trace Example</summary>

To better understand the notification system, `get_planning_env()`, and the evolution of MDPs, consider the following execution trace. Suppose we have an initial MDP $\mathcal{M}_0$ whose transition function is parametrized by parameter $\theta_0$, and NS-Gym is configured to update $\theta$ every two decision epochs.

| | $t_1$ | $t_2$ | $t_3$ | $t_4$ | $t_5$ | $t_6$ | $t_7$ | $t_8$ | $t_9$ | $t_{10}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **MDP** | $\mathcal{M}_0$ | $\mathcal{M}_0$ | $\mathcal{M}_1$ | $\mathcal{M}_1$ | $\mathcal{M}_2$ | $\mathcal{M}_2$ | $\mathcal{M}_3$ | $\mathcal{M}_3$ | $\mathcal{M}_4$ | $\mathcal{M}_4$ |
| $\theta$ | $\theta_0$ | $\theta_0$ | $\theta_1$ | $\theta_1$ | $\theta_2$ | $\theta_2$ | $\theta_3$ | $\theta_3$ | $\theta_4$ | $\theta_4$ |

At initialization ($t_0$), the agent always knows $\mathcal{M}_0$ and $\theta_0$. Consider the transition from $t_6$ to $t_7$.

If `change_notification == True` and `delta_change_notification == False`, the agent is notified that we have transitioned from $\mathcal{M}_2$ to $\mathcal{M}_3$ but **does not know** $\theta_3$.

If `change_notification == True` and `delta_change_notification == True`, the agent is notified that we have transitioned from $\mathcal{M}_2$ to $\mathcal{M}_3$ and **does know** $\theta_3$.

If `change_notification == False` (and by default `delta_change_notification == False`), the agent is not notified of any changes and only has information about $\mathcal{M}_0$ and $\theta_0$.

Suppose after transitioning from $t_6$ to $t_7$ we call `planning_env = ns_env.get_planning_env()`.

If `change_notification == True` and `delta_change_notification == False`, `planning_env` will be a stationary copy of $\mathcal{M}_0$ since we do not know $\theta_3$.

If `change_notification == True` and `delta_change_notification == True`, `planning_env` will be a stationary copy of $\mathcal{M}_3$ because we do know $\theta_3$.

If `change_notification == False` and `delta_change_notification == False`, `planning_env` will be a stationary copy of $\mathcal{M}_0$.
</details>

## 5. Build the wrapped environment

Now, we put the pieces we learned together into a single CartPole environment:

```python
import gymnasium as gym
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import OscillatingUpdate, RandomWalk
from ns_gym.wrappers import NSClassicControlWrapper


def make_env(**kwargs):
    change_notification = kwargs.get("change_notification", False)
    delta_change_notification = kwargs.get("delta_change_notification", False)

    base_env = gym.make("CartPole-v1")

    tunable_params = {
        "masscart": OscillatingUpdate(ContinuousScheduler()),
        "masspole": RandomWalk(PeriodicScheduler(period=5)),
    }

    return NSClassicControlWrapper(
        base_env,
        tunable_params,
        change_notification=change_notification,
        delta_change_notification=delta_change_notification,
    )
```

## 6. Register the environment

To reuse this environment elsewhere, register it with Gymnasium's registration API. Add a `register()` call in [src/AAMAS_Comp/\_\_init\_\_.py](src/AAMAS_Comp/__init__.py) pointing to your `make_env` function:

```python
register(
    id="MyCustomNSCartPole-v0",
    entry_point="AAMAS_Comp.examples.environments.my_custom_env:make_env",
    disable_env_checker=True,
    order_enforce=False,
)
```

You can then load it anywhere with `gym.make("MyCustomNSCartPole-v0")` and add the env ID to the `ENVIRONMENTS` dictionary in [evaluator.py](evaluator.py) to evaluate your agent on it!


Need help? Join in our [Office Hour sessions](../aamas2026_competition.md#office-hours)!

## Next step

Confident about your algorithm? Go to [Submit Your Agent](submission.md) and show us what you got!
