# Pendulum Environment

The Pendulum environment is configured to allow non-stationarity in physics parameters like pendulum length, mass and gravity.  See [Gymnasium documentation](https://gymnasium.farama.org/environments/classic_control/pendulum/) for more details on the environment. 


```{eval-rst}
.. figure:: /_static/images/pendulum.gif
   :width: 300px
   :align: center
   :alt: Pendulum Animation

   The Pendulum Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```   


## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `m` | Mass of the pendulum | 1.0 |
| `l` | Length of the pendulum   | 1.0 |
| `dt` | Time step between updates | 0.05 |
| `g` | Magnitude of gravitational acceleration | 10.0 |
