# FrozenLake Environment

The FrozenLake environment is a gridworld environment with a stochastic transition model. The agent must navigate from a starting point to a goal point while avoiding holes in the frozen lake. For each action taken by the agent,  agent will either move in the direction corresponding to the action taken or "slip" to one of two perpendicular directions with equal probability. See [Gymnasium documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) for more details on the environment.

```{eval-rst}
.. figure:: /_static/images/frozenlake_nonstationary.gif
   :width: 300px
   :align: center
   :alt: FrozenLake Animation  
    The FrozenLake Environment. 

```


## Tunable Parameters

In NS-Gym we can only update the "sliperryness" of the gridworld, which affects the probability of the agent's movement being altered to a different perpendicular direction.

| Parameter | Description | Default Value |
|----|------|------|
| `P` | Categorial distribution over next states for each action | Deterministic (no slip) [1.0, 0.0, 0.0] |

```{eval-rst}
.. important::
    By convention, the categorical distribution `P` is defined as `[p_intended, p_perpendicular_1, p_perpendicular_2]`, where `p_intended` is the probability of moving in the intended direction, and `p_perpendicular_1` and `p_perpendicular_2` are the probabilities of moving in the two perpendicular directions.




