# Ant Environment

The Ant environment is a high-dimensional continuous control task where a four-legged robot (ant) must learn to walk and navigate through its environment. The goal is to maximize forward velocity while maintaining stability. This environment is commonly used for benchmarking reinforcement learning algorithms. See the [Gymnasium documentation](https://gymnasium.farama.org/environments/mujoco/ant/) for more details.

```{eval-rst}
.. figure:: /_static/images/ant.gif
   :width: 300px
   :align: center
   :alt: Ant Animation

   The Ant Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```

## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Global gravitational acceleration | [ 0.    0.   -9.81] |
| `torso_mass` | Mass of the torso | 0.32724923474893675 |
| `floor_friction` | Friction coefficient of the floor | 1.0 |

**Note on `floor_friction`:** MuJoCo uses the larger of the two touching geoms' friction for a contact. The robot's own contact geoms have a friction of 1 here, so `floor_friction` only changes the dynamics when it is above 1. Lowering it below that value has no effect.
