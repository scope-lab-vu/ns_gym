# HalfCheetah Environment

The HalfCheetah environment is a high-dimensional continuous control task where a two-legged robot (half-cheetah) must learn to run forward as fast as possible. The goal is to maximize forward velocity while maintaining balance and stability. See the [Gymnasium documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) for more details.

```{eval-rst}
.. figure:: /_static/images/half_cheetah.gif
   :width: 300px
   :align: center
   :alt: HalfCheetah Animation

   The HalfCheetah Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```

## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Global gravitational acceleration | [ 0.    0.   -9.81] |
| `torso_mass` | Mass of the torso | 6.25020920502092 |
| `bthigh_mass` | Mass of the back thigh | 1.5435146443514645 |
| `bshin_mass` | Mass of the back shin | 1.5874476987447697 |
| `bfoot_mass` | Mass of the back foot | 1.0953974895397491 |
| `fthigh_mass` | Mass of the front thigh | 1.4380753138075317 |
| `fshin_mass` | Mass of the front shin | 1.200836820083682 |
| `ffeet_mass` | Mass of the front feet | 0.8845188284518829 |
| `floor_friction` | Friction coefficient of the floor | 0.4 |
| `bthigh_damping` | Back thigh joint damping | 6.0 |
| `bshin_damping` | Back shin joint damping | 4.5 |
| `bfoot_damping` | Back foot joint damping | 3.0 |
| `fthigh_damping` | Front thigh joint damping | 4.5 |
| `fshin_damping` | Front shin joint damping | 3.0 |
| `ffeet_damping` | Front feet joint damping | 1.5 |

