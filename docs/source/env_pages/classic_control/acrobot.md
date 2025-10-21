# Acrobot Environment

The Acrobot environment is configured to allow non-stationarity in physics parameters like link mass and link length. See [Gymnasium documentation](https://gymnasium.farama.org/environments/classic_control/acrobot/) for more details on the environment. 

```{eval-rst}
.. figure:: /_static/images/acrobot.gif
   :width: 300px
   :align: center
   :alt: Acrobot Animation

   The Acrobot Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```   

## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `dt` | Time between control updates | 0.2 |
| `LINK_LENGTH_1` | Length of the link connecting the first joint to the second joint | 1.0 |
| `LINK_LENGTH_2` | Length of the link only connected the second joint | 1.0 |
| `LINK_MASS_1` | Mass of the first link | 1.0 |
| `LINK_MASS_2` | Mass of the second link | 1.0 |
| `LINK_COM_POS_1` | Center of mass position of the first link | 0.5 |
| `LINK_COM_POS_2` | Center of mass position of the second link | 0.5 |
| `LINK_MOI` | Moment of inertia of the links | 1.0 |