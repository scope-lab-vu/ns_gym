# CliffWalking Environment

The CliffWalking environment is a stochastic gridworld environment where an agent must navigate from a start state to a goal state while avoiding a "cliff" region that incurs a large negative reward. There is no termination condition if the agent falls down the "cliff". See [Gymnasium documentation](https://gymnasium.farama.org/environments/toy_text/cliff_walking/) for more details on the environment.


## Tunable Parameters

In the CliffWalking environment we directly modify the "sliperryness" of the gridworld, which affects the probability of the agent's intended action being altered to a different action. By default, the environment is fully deterministic (i.e., no slip). As the sliperryness increases the probablity of moving in the intended direction decreases, while the probability of moving in the reverse or perpendicular directions increases by an equal amount. For example the initial categorical distribution over possible next states for action "right" is [1.0,0.0,0.0,0.0] (i.e., 100% chance of moving right). If we decrease chance of going in the intended direction by 0.4 NS-Gym updates categorical distribution as [0.6,0.1,0.1,0.1] (i.e., 60% chance of moving right, 20% chance of moving left, and 10% chance of moving up or down).


| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `P` | Categorial distribution over next states for each action | Deterministic (no slip) [1.0, 0.0, 0.0, 0.0] |

```{eval-rst}
.. important::
    By convention, the categorical distribution `P` is defined as `[p_intended, p_perpendicular_1, p_perpendicular_2, p_reverse]`, where `p_intended` is the probability of moving in the intended direction, and `p_perpendicular_1`, `p_perpendicular_2`, and `p_reverse` are the probabilities of moving in the two perpendicular directions and the reverse direction, respectively.
