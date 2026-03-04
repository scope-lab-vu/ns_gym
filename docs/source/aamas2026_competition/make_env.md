# Create a Custom Non-Stationary Environment

Welcome! This is the third tutorial in a series for the full submission workflow:

1.  [Environment Setup](setup.md): create your repository, configure Python, and build Docker images.
2.  [Build Your Agent](build_agents.md): implement a model-based or model-free agent and register it.
3. This tutorial: define non-stationarity with schedulers and update functions.
4. [Submit Your Agent](submission.md): run final checks and send your repository for evaluation.

This is the part you will leverage Ns-Gym. Let's get started!

> “The beginning of wisdom is to call things by their proper name.” – Confucius

## Overview

A custom NS-Gym environment has three parts:

1. Base Gymnasium environment
2. Scheduler (`when` changes happen)
3. Update function (`how` parameters change)

We can also design the agent's "notification level", described in section 6.

## 1. Pick a base environment and wrapper

Use the wrapper that matches the environment family:

- `FrozenLake-v1` -> `NSFrozenLakeWrapper`
- `CartPole-v1` -> `NSClassicControlWrapper`
- `Ant-v5` -> `NSMujocoWrapper`

## 2. Define change timing with schedulers

Common scheduler choices:

- `ContinuousScheduler()`
- `PeriodicScheduler(period=n)`
- `DiscreteScheduler({t1, t2, ...})`
- `RandomScheduler(p)`
- `MemorylessScheduler(p)`
- `CustomScheduler(fn)`

## 3. Define parameter updates

Attach an update function to each tunable parameter.

Examples:

- increment/decrement updates,
- random walk updates,
- bounded updates,
- distribution updates for stochastic environments.

## 4. Build the wrapped environment

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

## 5. Register the environment

Add a registration entry to `src/AAMAS_Comp/__init__.py`:

```python
register(
    id="MyCustomNSCartPole-v0",
    entry_point="AAMAS_Comp.examples.environments.my_custom_env:make_env",
    disable_env_checker=True,
    order_enforce=False,
)
```

Then instantiate it with:

```python
env = gym.make("MyCustomNSCartPole-v0")
```

## 6. Configure change notifications

- `change_notification=True`: tells the agent which parameters changed.
- `delta_change_notification=True`: also provides change magnitudes.

Observation fields may include:

- `state`
- `relative_time`
- `env_change`
- `delta_change`

## 7. Evaluate your custom environment

1. Add the environment ID to the `ENVIRONMENTS` dictionary in `evaluator.py`.
2. Run local evaluation:

```bash
uv run python evaluator.py
```

3. Run container evaluation:

```bash
docker compose run --rm test-submission
```

## Next step

Go to [Submit Your Agent](submission.md).
