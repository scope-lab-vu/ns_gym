# Pusher Environment

The Pusher environment is a high-dimensional continuous control task where a robotic arm must learn to push a block to a target location on a table. The goal is to minimize the distance between the block and the target while efficiently controlling the arm's movements. See the [Gymnasium documentation](https://gymnasium.farama.org/environments/mujoco/pusher/) for more details.

```{eval-rst}
.. figure:: /_static/images/pusher.gif
   :width: 300px
   :align: center
   :alt: Pusher Animation

   The Pusher Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```

## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Global gravitational acceleration | [0. 0. 0.] |
| `r_shoulder_pan_link_mass` | Mass of the right shoulder pan link | 7.293521504574065 |
| `r_shoulder_lift_link_mass` | Mass of the right shoulder lift link | 3.141592653589794 |
| `r_upper_arm_link_mass` | Mass of the right upper arm link | 1.6286016316209488 |
| `r_forearm_link_mass` | Mass of the right forearm link | 0.8427322293254622 |
| `r_shoulder_pan_joint_damping` | Damping of the right shoulder pan joint | 1.0 |
| `r_shoulder_lift_joint_damping` | Damping of the right shoulder lift joint | 1.0 |
| `r_elbow_flex_joint_damping` | Damping of the right elbow flex joint | 0.1 |
