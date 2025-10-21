# Swimmer Environment

The Swimmer environment is a high-dimensional continuous control task where a multi-segmented robotic swimmer must learn to propel itself forward in a fluid environment. The goal is to maximize forward velocity while maintaining stability and efficient movement. See the [Gymnasium documentation](https://gymnasium.farama.org/environments/mujoco/swimmer/) for more details.

```{eval-rst}
.. figure:: /_static/images/swimmer.gif
   :width: 300px
   :align: center
   :alt: Swimmer Animation

   The Swimmer Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```

## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Global gravitational acceleration | [ 0.    0.   -9.81] |
| `body_mid_mass` | Mass of the middle body segment | 35.604716740684324 |
| `geom_floor_friction` | Coefficient of friction for the floor | 1.0 |
