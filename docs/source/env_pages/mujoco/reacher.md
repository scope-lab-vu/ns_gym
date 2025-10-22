# Reacher Environment
The Reacher environment is a high-dimensional continuous control task where a two-jointed robotic arm must learn to reach a target location in a 2D plane. The goal is to minimize the distance between the arm's end effector and the target while efficiently controlling the arm's movements. See the [Gymnasium documentation](https://gymnasium.farama.org/environments/mujoco/reacher/) for more details.

```{eval-rst}
.. figure:: /_static/images/reacher.gif
   :width: 300px
   :align: center
   :alt: Reacher Animation

   The Reacher Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```

## Tunable Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Global gravitational acceleration | [ 0.    0.   -9.81] |
| `body0_mass` | Mass of the first body | 0.03560471674068433 |
| `body1_mass` | Mass of the second body | 0.03560471674068433 |
| `ground_friction` | Coefficient of friction for the ground | 1.0 |
| `joint0_damping` | Damping of the first joint | 1.0 |
| `joint1_damping` | Damping of the second joint | 1.0 |


