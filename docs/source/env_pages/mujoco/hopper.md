# Hopper Environment

The Hopper environment is a high-dimensional continuous control task where a one-legged robot (hopper) must learn to hop forward as quickly as possible while maintaining balance. The goal is to maximize forward velocity while preventing the hopper from falling over. See the [Gymnasium documentation](https://gymnasium.farama.org/environments/mujoco/hopper/) for more details.

```{eval-rst}
.. figure:: /_static/images/hopper.gif
   :width: 300px
   :align: center
   :alt: Hopper Animation

   The Hopper Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```

## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Global gravitational acceleration | [ 0.    0.   -9.81] |
| `torso_mass` | Mass of the torso | 3.6651914291880923 |
| `thigh_mass` | Mass of the thigh | 4.057890510886818 |
| `leg_mass` | Mass of the leg | 2.7813566959781637 |
| `foot_mass` | Mass of the foot | 5.31557476987393 |
| `floor_friction` | Friction coefficient of the floor | 1.0 |
| `thigh_joint_damping` | Damping coefficient of the thigh joint | 1.0 |
| `leg_joint_damping` | Damping coefficient of the leg joint | 1.0 |
| `foot_joint_damping` | Damping coefficient of the foot joint | 1.0 |

