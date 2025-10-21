# Humanoid Environment

The Humanoid environment is a high-dimensional continuous control task where a humanoid robot must learn to walk and navigate through its environment. The goal is to maximize forward velocity while maintaining balance and stability.  See the [Gymnasium documentation](https://gymnasium.farama.org/environments/mujoco/humanoid/) for more details.

```{eval-rst}
.. figure:: /_static/images/humanoid.gif
   :width: 300px
   :align: center
   :alt: Humanoid Animation

   The Humanoid Environment. GIF courtesy of the `Gymnasium Documentation <https://gymnasium.farama.org/>`_.
```

## Tunable Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `gravity` | Global gravitational acceleration | [ 0.    0.   -9.81] |
| `torso_mass` | Mass of the torso | 8.907462370478262 |
| `lwaist_mass` | Mass of the left waist | 2.261946710584651 |
| `pelvis_mass` | Mass of the pelvis | 6.616194128460103 |
| `right_thigh_mass` | Mass of the right thigh | 4.751750928806241 |
| `left_thigh_mass` | Mass of the left thigh | 4.751750928806241 |
| `right_shin_mass` | Mass of the right shin | 2.7556961671836424 |
| `left_shin_mass` | Mass of the left shin | 2.7556961671836424 |
| `right_foot_mass` | Mass of the right foot | 1.7671458676442586 |
| `left_foot_mass` | Mass of the left foot | 1.7671458676442586 |
| `right_upper_arm_mass` | Mass of the right upper arm | 1.6610804848382084 |
| `left_upper_arm_mass` | Mass of the left upper arm | 1.6610804848382084 |
| `right_lower_arm_mass` | Mass of the right lower arm | 1.2295401928310803 |
| `left_lower_arm_mass` | Mass of the left lower arm | 1.2295401928310803 |
| `floor_friction` | Friction coefficient of the floor | 1.0 |
| `right_knee_damping` | Damping coefficient of the right knee joint | 5.0 |
| `left_knee_damping` | Damping coefficient of the left knee joint | 5.0 |
| `right_elbow_damping` | Damping coefficient of the right elbow joint | 5.0 |
| `left_elbow_damping` | Damping coefficient of the left elbow joint | 1.0 |

