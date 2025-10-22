# Non-Stationary Bridge Environment


The non-stationary bridge environment is a gridworld setting where the agent must
navigate from the starting cell to one of two goal cells. The environment was originally introduced by (Lecarpentier and Rachelson [2019])[https://papers.nips.cc/paper_files/paper/2019/file/859b00aec8885efc83d1541b52a1220d-Paper.pdf]. To reach a goal cell, the agent must cross a “bridge” surrounded by terminal cells. The secondary goal cell is farther from the starting location but less risky because fewer holes surround it. Unlike the CliffWalking environment, which has a single global transition probability, the left and right halves of the Bridge map each have separate probability distributions.  NS-Gym allows for updates to just the left or right halves of the map or to the global value. Similar to the FrozenLake environment, if the agent moves in some direction, there is some probability that is moves in one of the perpendicular directions instead. The agent receives a +1 reward for reaching a goal cell, a -1 reward for falling into a hole, and a 0 reward otherwise. Our version of the non-stationary bridge environment is not included in the standard Gymnasium Python package.


## Tunable Parameters


| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `P` | Categorial distribution over next states for each action | Deterministic (no slip) [1.0, 0.0, 0.0] |
| `right_side` | Slipperiness of the right side of the bridge | [1.0, 0.0, 0.0] |
| `left_side` | Slipperiness of the left side of the bridge | [1.0, 0.0, 0.0] |

  By convention, the categorical distribution `P` is defined as [p_intended, p_perpendicular_1, p_perpendicular_2], where `p_intended` is the probability of moving in the intended direction, and `p_perpendicular_1` and `p_perpendicular_2` are the probabilities of moving in the two perpendicular directions.

```{eval-rst}
.. important::
    By convention, the categorical distribution `P` is defined as `[p_intended, p_perpendicular_1, p_perpendicular_2]`, where `p_intended` is the probability of moving in the intended direction, and `p_perpendicular_1` and `p_perpendicular_2` are the probabilities of moving in the two perpendicular directions.






