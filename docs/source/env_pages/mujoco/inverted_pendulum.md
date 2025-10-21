# Inverted Pendulum Environment

The Inverted Pendulum environment is a high-dimensional continuous control task where a cart must balance a pole upright while moving along a one-dimensional track. The goal is to keep the pole balanced while maximizing the distance traveled by the cart. See the [Gymnasium documentation](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) for more details.

```{eval-rst}
.. figure:: /_static/images/inverted_pendulum.gif
   :width: 300px
   :align: center
   :alt: Inverted Pendulum Animation

   The Inverted Pendulum Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```

## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Global gravitational acceleration | [ 0.    0.   -9.81] |
| `pole_mass` | Mass of the pole | 5.018591641363306 |
| `cart_mass` | Mass of the cart | 10.47197551196598 |
| `rail_friction` | Coefficient of friction for the rail | 1.0 |
