"""Clean import tests for ns_gym package.

Verifies that the package and its submodules can be imported without errors,
and that key classes/functions are accessible from their expected locations.
"""

import pytest


class TestTopLevelImport:
    """Test that the top-level ns_gym package imports cleanly."""

    def test_import_ns_gym(self):
        import ns_gym
        assert hasattr(ns_gym, "__version__")

    def test_version_is_string(self):
        import ns_gym
        assert isinstance(ns_gym.__version__, str)


class TestSubmoduleImports:
    """Test that all submodules import without errors."""

    @pytest.mark.parametrize("submodule", [
        "ns_gym.base",
        "ns_gym.schedulers",
        "ns_gym.update_functions",
        "ns_gym.wrappers",
        "ns_gym.utils",
        "ns_gym.evaluate",
        "ns_gym.benchmark_algorithms",
        "ns_gym.context_switching",
        "ns_gym.envs",
    ])
    def test_import_submodule(self, submodule):
        import importlib
        mod = importlib.import_module(submodule)
        assert mod is not None


class TestBaseClasses:
    """Test that key base classes are importable."""

    def test_import_ns_wrapper(self):
        from ns_gym.base import NSWrapper
        assert NSWrapper is not None

    def test_import_update_fn(self):
        from ns_gym.base import UpdateFn
        assert UpdateFn is not None

    def test_import_update_distribution_fn(self):
        from ns_gym.base import UpdateDistributionFn
        assert UpdateDistributionFn is not None

    def test_import_scheduler(self):
        from ns_gym.base import Scheduler
        assert Scheduler is not None


class TestSchedulers:
    """Test that scheduler classes are importable."""

    def test_import_continuous_scheduler(self):
        from ns_gym.schedulers import ContinuousScheduler
        assert ContinuousScheduler is not None

    def test_import_discrete_scheduler(self):
        from ns_gym.schedulers import DiscreteScheduler
        assert DiscreteScheduler is not None


class TestUpdateFunctions:
    """Test that update function classes are importable."""

    def test_import_random_walk(self):
        from ns_gym.update_functions import RandomWalk
        assert RandomWalk is not None

    def test_import_random_categorical(self):
        from ns_gym.update_functions import RandomCategorical
        assert RandomCategorical is not None

    def test_import_linear_interpolation(self):
        from ns_gym.update_functions import LinearInterpolation
        assert LinearInterpolation is not None


class TestWrappers:
    """Test that wrapper classes are importable."""

    def test_import_classic_control_wrapper(self):
        from ns_gym.wrappers import NSClassicControlWrapper
        assert NSClassicControlWrapper is not None

    def test_import_mujoco_wrapper(self):
        from ns_gym.wrappers import MujocoWrapper
        assert MujocoWrapper is not None

    def test_import_frozen_lake_wrapper(self):
        from ns_gym.wrappers import NSFrozenLakeWrapper
        assert NSFrozenLakeWrapper is not None

    def test_import_cliff_walking_wrapper(self):
        from ns_gym.wrappers import NSCliffWalkingWrapper
        assert NSCliffWalkingWrapper is not None

    def test_import_bridge_wrapper(self):
        from ns_gym.wrappers import NSBridgeWrapper
        assert NSBridgeWrapper is not None

    def test_import_pursuit_evasion_wrapper(self):
        from ns_gym.wrappers import PursuitEvasionWrapper
        assert PursuitEvasionWrapper is not None


class TestBenchmarkAlgorithms:
    """Test that benchmark algorithm classes are importable."""

    def test_import_mcts(self):
        from ns_gym.benchmark_algorithms import MCTS
        assert MCTS is not None

    def test_import_pamcts(self):
        from ns_gym.benchmark_algorithms import PAMCTS
        assert PAMCTS is not None

    def test_import_dqn(self):
        from ns_gym.benchmark_algorithms import DQN, DQNAgent, train_ddqn
        assert DQN is not None
        assert DQNAgent is not None
        assert train_ddqn is not None

    def test_import_ppo(self):
        from ns_gym.benchmark_algorithms import PPO, PPOActor, PPOCritic
        assert PPO is not None
        assert PPOActor is not None
        assert PPOCritic is not None

    def test_import_alphazero(self):
        from ns_gym.benchmark_algorithms import AlphaZeroAgent, AlphaZeroNetwork
        assert AlphaZeroAgent is not None
        assert AlphaZeroNetwork is not None

    def test_import_ddpg(self):
        from ns_gym.benchmark_algorithms import DDPG
        assert DDPG is not None


class TestLazyLoading:
    """Test that heavy dependencies are not loaded eagerly."""

    def test_ns_gym_import_does_not_load_torch(self):
        """Verify that 'import ns_gym' does not eagerly load torch."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; import ns_gym; print('torch' not in sys.modules)"],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "True", (
            "torch should not be loaded by 'import ns_gym'"
        )

    def test_lazy_submodules_accessible(self):
        """Verify lazy-loaded submodules are still accessible on demand."""
        import ns_gym

        assert hasattr(ns_gym, "benchmark_algorithms") or callable(getattr(type(ns_gym), '__getattr__', None))
        # Actually access the lazy module to confirm it loads
        mod = ns_gym.benchmark_algorithms
        assert mod is not None
