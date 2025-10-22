# MountainCar Environment

The MountainCar environment is a classic control problem where an underpowered car must drive up a steep hill. The car cannot reach the top of the hill directly due to insufficient engine power, so it must build momentum by oscillating back and forth. Both the discrete and continuous action spaces version of the environment are supported. See the [Gymnasium documentation](https://gymnasium.farama.org/environments/classic_control/mountain_car/) for more details.

```{eval-rst}
.. figure:: /_static/images/mountain_car.gif
   :width: 300px
   :align: center
   :alt: MountainCar Animation

   The MountainCar Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```   


## Tunable Parameters

### Discrete Action Space

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Magnitude of gravitational acceleration | 0.0025 |
| `force` | Magnitude of the force applied to the car | 0.001 |


### Continuous Action Space

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `power` | Magnitude of the power applied to the car | 0.0015 |