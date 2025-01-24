import unittest
import ns_gym
import gymnasium as gym
from ns_gym.base import Evaluator
from ns_gym.eval.metrics import *


class BaseTestEvaluator(unittest.TestCase):
    """
    Base test class for Evaluators. Handles common setup and test logic.
    """

    def setUp(self):
        # Create the environments
        self.env1 = self.create_ns_env(0.5)
        self.env2 = self.create_ns_env(0.0)

    def create_ns_env(self, update_value):
        """
        Helper function to create non-stationary environments.
        """
        env = gym.make("FrozenLake-v1")
        scheduler = ns_gym.schedulers.ContinuousScheduler(start=0, end=0)
        update_fn = ns_gym.update_functions.DistributionDecrementUpdate(scheduler, update_value)
        parameter_map = {"P": update_fn}
        ns_env = ns_gym.wrappers.NSFrozenLakeWrapper(env, tunable_params=parameter_map)
        ns_env.reset()
        a = ns_env.action_space.sample()
        ns_env.step(a)
        return ns_env
    
    def create_cliffwalking_env(self, update_value):
        env = gym.make("CliffWalking-v0")
        scheduler = ns_gym.schedulers.ContinuousScheduler(start=0, end=0)
        update_fn = ns_gym.update_functions.DistributionDecrementUpdate(scheduler, update_value)
        parameter_map = {"P": update_fn}
        ns_env = ns_gym.wrappers.NSCliffWalkingWrapper(env, tunable_params=parameter_map)
        ns_env.reset()
        a = ns_env.action_space.sample()
        ns_env.step(a)
        return ns_env
    
    def create_bridge_env(self, update_value):
        env = gym.make("ns_gym/Bridge-v0")
        scheduler = ns_gym.schedulers.ContinuousScheduler(start=0, end=0)
        update_fn = ns_gym.update_functions.DistributionDecrementUpdate(scheduler, update_value)
        parameter_map = {"_P": update_fn}
        ns_env = ns_gym.wrappers.NSBridgeWrapper(env, tunable_params=parameter_map)
        ns_env.reset()
        a = ns_env.action_space.sample()
        ns_env.step(a)
        return ns_env
    
    def create_pendulum_env(self, update_value):    
        env = gym.make("Pendulum-v0")
        scheduler = ns_gym.schedulers.ContinuousScheduler(start=0, end=0)
        update_fn = ns_gym.update_functions.DecrementUpdate(scheduler, update_value)
        parameter_map = {"m": update_fn}
        ns_env = ns_gym.wrappers.NSPendulumWrapper(env, tunable_params=parameter_map)
        ns_env.reset()
        a = ns_env.action_space.sample()
        ns_env.step(a)
        return ns_env

    def evaluate(self, evaluator_cls):
        """
        Common evaluation logic for all evaluator subclasses.
        """
        evaluator = evaluator_cls()
        try:
            result = evaluator.evaluate(self.env1, self.env2)
            self.assertIsInstance(result, (int, float), "Result should be numeric")
            self.assertGreaterEqual(result, 0.0, "Result should be non-negative")

        except Exception as e:
            if isinstance(e, NotImplementedError):
                self.skipTest(f"{evaluator_cls.__name__} has not implemented 'evaluate' yet.")
            else:
                raise e
            


# Define explicit test cases for each Evaluator subclass
class TestEnsembleMetric(BaseTestEvaluator):
    def test_load_agents(self):
        from ns_gym.eval.metrics import EnsembleMetric

        env = self.create_pendulum_env(0.5)

        evaluator = EnsembleMetric(env, 5)

    def test_pendulum(self):
        from ns_gym.eval.metrics import EnsembleMetric
        evaluator = EnsembleMetric()
        env = self.create_pendulum_env(0.5)
        result = evaluator(env)

        


    

class TestPAMCTSBound(BaseTestEvaluator):
    def test_evaluate(self):
        from ns_gym.eval.metrics import PAMCTS_Bound
        self.evaluate(PAMCTS_Bound)

class TestTSMDPBound(BaseTestEvaluator):
    def test_evaluate(self):
        from ns_gym.eval.metrics import TSMDP_Bound
        self.evaluate(TSMDP_Bound)

    def test_frozenlake(self):
        from ns_gym.eval.metrics import TSMDP_Bound
        evaluator = TSMDP_Bound()

        self.assertIsInstance(self.env1.observation_space, gym.spaces.Discrete)
        result = evaluator.evaluate(self.env1, self.env2)
        self.assertEqual(result, 1-(0.5/2))

    def test_cliffwalking(self):
        from ns_gym.eval.metrics import TSMDP_Bound
        evaluator = TSMDP_Bound()

        self.assertIsInstance(self.env1.observation_space, gym.spaces.Discrete)
        result = evaluator.evaluate(self.create_cliffwalking_env(0.5), self.create_cliffwalking_env(0.0))
        self.assertEqual(result, 1-(0.5/3))

    # def test_bridge(self):
    #     from ns_gym.eval.metrics import TSMDP_Bound
    #     evaluator = TSMDP_Bound()

    #     self.assertIsInstance(self.env1.observation_space, gym.spaces.Discrete)
    #     result = evaluator.evaluate(self.create_bridge_env(0.5), self.create_bridge_env(0.0))
    #     self.assertEqual(result, 1-(0.5/3))

if __name__ == "__main__":
    unittest.main()