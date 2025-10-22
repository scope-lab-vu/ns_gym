# Inverted Double Pendulum Environment

The Inverted Double Pendulum environment is a high-dimensional continuous control task where a cart must balance a double pendulum upright while moving along a one-dimensional track. The goal is to keep the pendulum balanced while maximizing the distance traveled by the cart. See the [Gymnasium documentation](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) for more details.

```{eval-rst}
.. figure:: /_static/images/inverted_double_pendulum.gif
   :width: 300px
   :align: center
   :alt: Inverted Double Pendulum Animation

   The Inverted Double Pendulum Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```

## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Global gravitational acceleration | [ 1.00e-05  0.00e+00 -9.81e+00] |
| `cart_mass` | Mass of the cart | 10.47197551196598 |
| `pole1_mass` | Mass of the first pole | 4.198738581522758 |
| `pole2_mass` | Mass of the second pole | 4.198738581522758 |
| `floor_friction` | Coefficient of friction for the floor | 1.0 |
| `slider_damping` | Damping of the slider joint | 0.05 |
| `hinge1_damping` | Damping of the first hinge joint | 0.05 |
| `hinge2_damping` | Damping of the second hinge joint | 0.05 |

