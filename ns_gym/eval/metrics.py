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
import importlib
import ns_gym
import ns_gym.schedulers
import ns_gym.update_functions
import ns_gym.wrappers


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
    def __init__(self,agents={}) -> None:
        """
        Args:
            agents (dict): A dictionary of agents to evaluate. The keys are the agent names and the values are the agent objects. Defaults to an empty dictionary.
        """
        super().__init__()
        self.agents = agents
    
    def evaluate(self,env,M=100,include_MCTS=False,include_RL=True,include_AlphaZero=False,verbose=True):
        """Evaluate the difficulty of a particular NS-MDP by comparing the mean reward over an ensemble of agents.
        NS-Gym uses the following procedure to evaluate the difficulty of a particular NS-MDP:

        For a particular NS-MDP, NS-Gym will look too see if there are saved agents in the directory. By default we will evaluate using StableBaseline3 RL agents.
        If there are no saved agents (say for custom environments), you will be prompted to train the agents.

        Args:
            env (gym.Env): The non-stationary environment to evaluate
            M (int): The number of episodes to run. Defaults to 100.
            include_MCTS (bool): Whether to include the MCTS agent in the ensemble. Defaults to False.
            include_RL (bool): Whether to include the RL agents in the ensemble. Defaults to True.
            include_AlphaZero (bool): Whether to include the AlphaZero agent in the ensemble. Defaults to False.
            verbose (bool): Whether to print the results of the evaluation. Defaults to True.

        Returns:
            ensemble_performance (float): The mean reward over the ensemble of agents
            performance (dict): A dictionary of the performance of each agent in the ensemble

        """

        agent_list = self._load_agents(env) # returns a list of agent names, agent objects stored in self.agents

        performance = {}

        if not agent_list:
            raise ValueError("No agents found in the evaluation_model_weights directory. Please train some agents first.")
        
        base_ensebleperformance, base_performance = self._evaluate_stable_baselines(env,agent_list,M)

        for i,agent_name in enumerate(agent_list):
            agent = self.agents[agent_name]
            performance[agent_name] = []
            for ep in range(M):

                total_reward = 0
                obs,info = env.reset()
                obs,_ = ns_gym.utils.type_mismatch_checker(obs,None)

                done = False
                truncated = False

                total_reward = 0
                while not (done or truncated):
                    # ns_gym.utils.neural_network_checker(self.agents[i].device,obs)
                    action = agent.act(obs)
                    action = ns_gym.eval.action_type_checker(action)
                    obs, reward, done, truncated,info = env.step(action)
                    obs,reward = ns_gym.utils.type_mismatch_checker(obs,reward)
                    total_reward += reward

                performance[agent_name].append(total_reward)
            
            performance[agent_name] = sum(performance[agent_name])/M

        ensemble_performance = sum(performance.values())/len(performance)

        regret = ensemble_performance - base_ensebleperformance

        agent_wise_regret = {agent_name:performance[agent_name] - base_performance[agent_name] for agent_name in agent_list}

        if verbose:
            self._print_results(ensemble_performance, performance)

        return ensemble_performance, performance    
                

    def _load_agents(self,env):
        """
        Load agents from the agent_paths
        """

        if self.agents:
            return list(self.agents.keys())
        
        else:
            env_name = env.unwrapped.__class__.__name__
            eval_dir = pathlib.Path(__file__).parent / "evaluation_model_weights" / env_name
            agent_paths = os.listdir(eval_dir) # this grabs the available agents for the environment (it is a list of paths to the agents)

            try:
                import stable_baselines3
            except:
                raise ImportError("Stable Baselines 3 is required to load agents")
            
            loaded_agents = []
            for agent in agent_paths:

                agent_dir = eval_dir / agent

                model  = getattr(stable_baselines3,agent)
                weights = [x for x in agent_dir.iterdir() if x.suffix.lower()==".zip"]

                if not weights:
                    warnings.warn(f"No weights found for {agent}. Skipping...")
                    continue

                elif len(weights) > 1:
                    warnings.warn(f"Multiple weights found for {agent}. Using the first one.")
                    
                model = model.load(weights[0])

                wrapped_model = ns_gym.base.StableBaselineWrapper(model)  

                loaded_agents.append(agent)  

                self.agents[agent]=(wrapped_model)

            return loaded_agents
        

    def _evaluate_stable_baselines(self,env,agent_list,M):
        """
        Evaluates the baseline_performance of the environment on default environments.
        """

        env_name = env.unwrapped.spec.id

        stationary_env = gym.make(env_name)

        performance = {agent_name:[] for agent_name in agent_list}
        
        for i,agent_name in enumerate(agent_list):
            agent = self.agents[agent_name]

            for ep in range(M):
                
                obs,_ = stationary_env.reset()
                done = False
                truncated = False
                total_reward = 0
                while not (done or truncated):
                    action = agent.act(obs)
                    obs, reward, done, truncated, info = stationary_env.step(action)
                    total_reward += reward

                performance[agent_name].append(total_reward)

            performance[agent_name] = sum(performance[agent_name])/M

        base_ensemble_performance = sum(performance.values())/len(performance)

        return base_ensemble_performance, performance        

        
    def _print_results(self, ensemble_performance, performance_dict):
        """
        Print the results of the evaluation in a structured format.

        Args:
            ensemble_performance (float): The performance metric for the ensemble.
            performance_dict (dict): A dictionary where keys are agent names and 
                                    values are their corresponding performance metrics.
        """
        print("=" * 40)
        print("Evaluation Results")
        print("=" * 40)
        print(f"Ensemble Regret: {ensemble_performance}\n")
        print("Agent Regret:")
        for agent, performance in performance_dict.items():
            print(f"  - {agent}: {performance}")
        print("=" * 40)



                    
 
class PAMCTS_Bound(ComparativeEvaluator):
    def __init__(self):
        super().__init__()
    
    def evaluate(self,env_1, env_2,verbose=True):
        """
        Evaluate the difficulty of a transition between two environments.
        For a particular state s, $forall a in A: |P_t(s'|s,a) - P_0(s'|a,s)|_{infty}$ 

        Args:
            env_1 (gym.Env): The original environment
            env_2 (gym.Env): The new environment
            verbose (bool): Whether to print the results of the evaluation. Defaults to True.

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

                if verbose:
                    self._print_results(max_diff)

                return max_diff
            
            except Exception as e:
                warnings.warn("This method only works for environments that have a transition matrix P[s][a][s_prime] that is a dictionary of dictionaries that maps to the probability of transitioning to state s_prime given state s and action a")


        elif self.space_type == "Box" and self.action_type == "Discrete":
            raise NotImplementedError
        
        elif self.space_type == "Discrete" and self.action_type == "Box":
            raise NotImplementedError
        
        else:
            raise ValueError("Observation space must be either Box or Discrete")
        

    def _print_results(max_diff):
        print("=" * 40)
        print("Evaluation Results")
        print("=" * 40)
        print(f"PAMCTS-Bound: {max_diff}")
        print("=" * 40)

        



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
    



if __name__ == "__main__":
    import ns_gym
    import gymnasium as gym


  
    env = gym.make('FrozenLake-v1',render_mode="rgb_array",max_episode_steps=50)
    scheduler = ns_gym.schedulers.DiscreteScheduler({1})
    update_function = ns_gym.update_functions.DistributionDecrementUpdate(scheduler=scheduler,k = 0.5)
    param = "P"
    params = {param:update_function}
    ns_env_1 = ns_gym.wrappers.NSFrozenLakeWrapper(env, params,initial_prob_dist=[1,0,0])

    ns_env_1.reset()

    ns_env_1.step(0)


    env = gym.make('FrozenLake-v1',render_mode="rgb_array",max_episode_steps=50)
    scheduler = ns_gym.schedulers.DiscreteScheduler({1})
    update_function = ns_gym.update_functions.DistributionDecrementUpdate(scheduler=scheduler,k = 0.5)
    params = {param:update_function}
    ns_env_2 = ns_gym.wrappers.NSFrozenLakeWrapper(env, params,initial_prob_dist=[1,0,0])

    evaluator = PAMCTS_Bound()

    bound = evaluator.evaluate(ns_env_1,ns_env_2)




