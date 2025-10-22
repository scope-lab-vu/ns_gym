# CartPole Environment

The CartPole environment is configured to allow non-stationarity in physics parameters like pole mass and gravity. See [Gymnasium documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/) for more details on the environment. 

```{eval-rst}
.. figure:: /_static/images/cartpole_temp.gif
   :width: 300px
   :align: center
   :alt: CartPole Animation

   The CartPole Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```   
See below for tunable parameters that can be modified to introduce non-stationarity.

## Tunable Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Magnitude of gravitational acceleration | 9.8 |
| `masscart` | Mass of the cart | 1.0 |
| `masspole` | Mass of the pole | 0.1 |
| `force_mag` | Magnitude of the force applied to the cart | 10.0 |
| `tau` | Time interval for state updates | 0.02 |
| `length` | Length of the pole | 0.5 |