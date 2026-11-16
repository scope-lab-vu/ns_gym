# Changelog

## Unreleased

### Fixed

Environments and wrappers
- FrozenLake and CliffWalking: after `reset()`, an episode no longer starts with the previous episode's transition table (#44).
- FrozenLake: the planning env from `get_planning_env()` is built on the real env's map instead of the default 4x4 map (#46).
- MuJoCo: `floor_friction` is accepted for Ant, Humanoid and HumanoidStandup, as documented (#45). The wrapper's allow-list and its getters/setters now come from one table (`base.MUJOCO_GETTERS`).
- MuJoCo: `reset()` restores the parameters before the base env builds its observation, so Humanoid's reset observation no longer carries the previous episode's body masses.
- MuJoCo: wrapping an env no longer changes its rewards. Ant and Humanoid previously paid slightly different rewards than the unwrapped env, and than their own planning envs, even when no parameter changed. States were never affected.
- Bridge: slips are drawn from the env's own generator, so `reset(seed=...)` makes episodes reproducible.
- Bridge: `NSBridgeWrapper.reset` no longer discards the seeding of its update functions.
- CartPole: `total_mass` and `polemass_length` are kept consistent with `masspole` after `reset()` and in planning envs created without change notification.
- Acrobot: `LINK_LENGTH_2` can no longer shrink below `LINK_COM_POS_2`.
- Planning envs keep the base env's `gym.make` arguments (for example `gym.make("Pendulum-v1", g=3.0)`), and the CliffWalking planning env keeps `modified_rewards` and `terminal_cliff`.
- MountainCar: removed a `UserWarning` that was emitted on every `deepcopy`.
- `freeze()` now stops parameter updates, as its documentation says. Previously it only hid the change notifications.
- Removed the registration of `ns_gym/VehicleTracking-v0`, whose module is no longer part of the package.

Schedulers and update functions
- `reset(seed=...)` now also seeds a random scheduler, and unseeded episodes draw a new schedule. `RandomScheduler`, `MemorylessScheduler` and `DecayingProbabilityScheduler` previously replayed the same change times in every episode.
- `MemorylessScheduler` with `start > 0` now fires; its first event is counted from `start`.
- `ExponentialDecay` follows its documented formula `Y_0 * exp(-lambda * t)` instead of compounding the decay on every call.
- `DistributionIncrementUpdate` and `DistributionDecrementUpdate` stay valid distributions for a negative `k`.
- `StepWiseUpdate` and `DistributionStepWiseUpdate` report no change once their list is exhausted, and no longer empty the list passed to them.
- `BudgetBoundedIncrement` can be constructed and enforces its budget.
- `LCBoundedDistrubutionUpdate` is seeded by `reset(seed=...)`.

Benchmark algorithms
- MCTS: backpropagation discounts once per decision and counts rewards collected inside the tree. Q-value estimates change as a result.
- MCTS: `search()` no longer raises `KeyError` when there are fewer simulations than actions.
- DDPG: `train()` no longer fails on an undefined variable, and stops after `num_episodes`.

### Changed
- `LCBoundedDistrubutionUpdate` moves toward one candidate distribution, up to the Lipschitz bound, instead of rejection sampling. Updates are much faster; the sequence of distributions for a given seed is different.
- Docs: removed parameters that have no physical effect and were never accepted (`rail_friction` on InvertedPendulum, `floor_friction` on InvertedDoublePendulum, `geom_floor_friction` and `gravity` on Swimmer, `ground_friction` and `gravity` on Reacher), corrected the Bridge parameter names to `P_left` / `P_right`, and noted that `floor_friction` only has an effect above the robot's own contact friction.

## Version 1.0.0 - 2025-21-10

### Added/Changed
- Changed observation interface from Python dataclass to dictionary for better compatibility with RL libraries.
- Added Sphinx documentation for the entire codebase.
- Added more comprehensive unit tests for all modules.
- Added MuJoCo environment support.
- Generally cleaned up and refactored code for better readability and maintainability.
- Updated README.
- Began using UV for package management.
- Added VehicleTrackingEnv environment.


