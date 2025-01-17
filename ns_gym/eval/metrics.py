from ns_gym.base import Evaluator
import scipy.integrate 
import warnings
import itertools
from abc import ABC
from typing import Type
from gymnasium import Env
import gymnasium as gym
import os 
import pathlib


class ComparativeEvaluator(Evaluator):
    """Superclass for evaluators that compare two environments. Handles checking that the environments are the same, etc 
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def evaluate(self, env_1: Type[Env], env_2: Type[Env],*args,**kwargs) -> float:
        assert env_1.unwrapped.__class__.__name__ == env_2.unwrapped.__class__.__name__, "Environments must be the same"
        assert env_1.observation_space == env_2.observation_space, "Observation spaces must be the same"
        assert env_1.action_space == env_2.action_space, "Action spaces must be the same"

        # loop through supported observation and action spaces...s

        # It may be the case for continuous action spaces that we have to simple discritize it ... 

       # Check observation space
        assert isinstance(env_1.observation_space, (gym.spaces.Box, gym.spaces.Discrete)), \
            "Unsupported observation space"
        assert isinstance(env_2.observation_space, (gym.spaces.Box, gym.spaces.Discrete)), \
            "Unsupported observation space"

        # Check action spacel

        self.space_type = env_1.observation_space.__class__.__name__
        self.action_type = env_1.action_space.__class__.__name__

    def __call__(self):
        return self.evaluate()

class EnsembleMetric(Evaluator):
    """
    Evaluates the difficulty of an NS-MDP by comparing mean reward over an ensemble of agents.
    """
    def __init__(self,agents=[]) -> None:
        super().__init__()
        self.agents = agents

    @staticmethod       
    def evaluate(self,env,M,include_MCTS=False,include_RL=True,include_AlphaZero=False):
        """Evaluate the difficulty of a particular NS-MDP by comparing the mean reward over an ensemble of agents.
        NS-Gym uses the following procedure to evaluate the difficulty of a particular NS-MDP:

        For a particular NS-MDP, NS-Gym will look too see if there are saved agents in the directory. By default we will evaluate using StableBaseline3 RL agents.
        If there are no saved agents (say for custom environments), you will be prompted to train the agents.

        Args:
            M (int): The number of episodes to run
            include_MCTS (bool): Whether to include the MCTS agent in the ensemble. 
        """

        agent_list = self.__load_agents(env)

        for agent in agent_list:
            pass

    def __load_agents(self,env):
        """
        Load agents from the agent_paths
        """

        if self.agents:
            return self.agents
        
        else:
            env_name = env.unwrapped.__class__.__name__
            agent_paths = os.listdir(pathlib.Path(__file__).parent / "evaluation_model_weights" / env_name)
            agent_list = []




class PAMCTS_Bound(ComparativeEvaluator):
    """Evaluates the difficulty of a NS-MDP Tranisition as a transition-bounded non-stationary Markov decision porcess (T-NSMDP)

    When the environment undergoes some change between time steps 0 and t, a T-NSMDP assumes 

    $$
    \forall s,a: \sum_{s'\in S}|P_t(s'|s,a) - P_0(s'|s,a)| \leq \eta
    $$

    Where $t\in \mathcal{T}$ is some point in time after the original policy was learned, and \eta is some scalar bound. 

    Of course if the environment is continious we instead integrate over the state space.
    """

    def __init__(self):
        super().__init__()

    def evaluate(self,env_1, env_2):
        """
        Evaluate the difficulty of a transition between two environments.

        TODO 
        """
        raise NotImplementedError

 
class TSMDP_Bound(ComparativeEvaluator):
    def __init__(self):
        super().__init__()
    
    def evaluate(self,env_1, env_2):
        """
        Evaluate the difficulty of a transition between two environments.
        For a particular state s, $\forall a \in A: |P_t(s'|s,a) - P_0(s'|a,s)|_{\infty}$ 

        Args:
            env_1 (gym.Env): The original environment
            env_2 (gym.Env): The new environment

        Returns:
            float: The maximum difference between the transition probabilities of the two environments
        """

        super().evaluate(env_1,env_2)

        if self.space_type == 'Box' and self.action_type == 'Box':
            raise NotImplementedError
        
        elif self.space_type == "Discrete" and self.action_type == "Discrete":

            # Right now we are assuming that the state space is discrete and the action space is discrete
            # Also assuming that there exists a transition matrix P[s_prime][s][a] that is a dictionary of dictionaries that maps to the probability of transitioning to state s_prime given state s and action a
            # This should work for environments inlcuded in the ns_gym package though it may not work for custom environments that do not have this structure...
            # Essesntially there needs to be a standardized way to access the transition probabilities of the environment...

            try:
                num_states = env_1.observation_space.n
                num_actions = env_1.action_space.n

                # P is a table of transition probabilities: keys are states. Then for each state, keys are actions, and values are dictionaries of next states and probabilities. P[s_prime][s][a] = P(s_prime | s, a)
                P1 = env_1.unwrapped.P
                P2 = env_2.unwrapped.P

                max_diff = 0
                for s in range(num_states):
                    for a in range(num_actions):
                        for s_prime_1,s_prime_2 in itertools.product([x for x in range(len(P1[s][a]))],repeat=2):
                            max_diff = max(max_diff, abs(P1[s][a][s_prime_1][0] - P2[s][a][s_prime_2][0])) # From state s with action a, what is the probability of transitioning to state s_prime

                return max_diff
            
            except Exception as e:
                warnings.warn("This method only works for environments that have a transition matrix P[s][a][s_prime] that is a dictionary of dictionaries that maps to the probability of transitioning to state s_prime given state s and action a")


        elif self.space_type == "Box" and self.action_type == "Discrete":
            raise NotImplementedError
        
        elif self.space_type == "Discrete" and self.action_type == "Box":
            raise NotImplementedError
        
        else:
            raise ValueError("Observation space must be either Box or Discrete")

class BIBO_Stablilty(Evaluator):
    
    def __init__(self):
        super().__init__()

    def evaluate(self,env1,env2):
        """
        Evaluate the stability of the environment.
        """
        raise NotImplementedError
    
class LyapunovStability(Evaluator):
    
    def __init__(self):
        super().__init__()

    def evaluate(self,env1,env2):
        """
        Evaluate the stability of the environment.
        """
        raise NotImplementedError
    




        

