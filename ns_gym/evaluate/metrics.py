from ns_gym.base import Evaluator
import warnings
from typing import Type
from gymnasium import Env
import gymnasium as gym
import os
import pathlib
import ns_gym
import ns_gym.schedulers
import ns_gym.update_functions
import ns_gym.wrappers
import torch


class ComparativeEvaluator(Evaluator):
    """Superclass for evaluators that compare two environments. Handles checking that the environments are the same, etc"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def evaluate(self, env_1: Type[Env], env_2: Type[Env], *args, **kwargs) -> float:
        assert (
            env_1.unwrapped.__class__.__name__ == env_2.unwrapped.__class__.__name__
        ), "Environments must be the same"
        assert env_1.unwrapped.observation_space == env_2.unwrapped.observation_space, (
            "Observation spaces must be the same"
        )
        assert env_1.unwrapped.action_space == env_2.unwrapped.action_space, (
            "Action spaces must be the same"
        )

        assert isinstance(
            env_1.unwrapped.observation_space, (gym.spaces.Box, gym.spaces.Discrete)
        ), "Unsupported observation space"
        assert isinstance(
            env_2.unwrapped.observation_space, (gym.spaces.Box, gym.spaces.Discrete)
        ), "Unsupported observation space"

        # Check action spacel

        self.space_type = env_1.unwrapped.observation_space.__class__.__name__
        self.action_type = env_1.unwrapped.action_space.__class__.__name__

    def __call__(self):
        return self.evaluate()


class EnsembleMetric(Evaluator):
    """
    Evaluates the difficulty of an NS-MDP by comparing mean reward over an ensemble of agents.

    Args:
        agents (dict): A dictionary of agents to evaluate. The keys are the agent names and the values are the agent objects. Defaults to an empty dictionary.
    """

    def __init__(self, agents={}) -> None:
        super().__init__()
        self.agents = agents

    def evaluate(
        self,
        env,
        M=100,
        include_MCTS=False,
        include_RL=True,
        include_AlphaZero=False,
        verbose=True,
    ):
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

        agent_list = self._load_agents(
            env
        )  # returns a list of agent names, agent objects stored in self.agents

        performance = {}

        if not agent_list:
            raise ValueError(
                "No agents found in the evaluation_model_weights directory. Please train some agents first."
            )

        base_ensebleperformance, base_performance = self._evaluate_stable_baselines(
            env, agent_list, M
        )

        for i, agent_name in enumerate(agent_list):
            agent = self.agents[agent_name]
            performance[agent_name] = []
            for ep in range(M):
                total_reward = 0
                obs, info = env.reset()
                obs, _ = ns_gym.utils.type_mismatch_checker(obs, None)

                done = False
                truncated = False

                total_reward = 0
                while not (done or truncated):
                    # ns_gym.utils.neural_network_checker(self.agents[i].device,obs)
                    action = agent.act(obs)
                    action = ns_gym.evaluate.action_type_checker(action)
                    obs, reward, done, truncated, info = env.step(action)
                    obs, reward = ns_gym.utils.type_mismatch_checker(obs, reward)
                    total_reward += reward

                performance[agent_name].append(total_reward)

            performance[agent_name] = sum(performance[agent_name]) / M

        ensemble_performance = sum(performance.values()) / len(performance)

        if verbose:
            self._print_results(ensemble_performance, performance)

        return ensemble_performance, performance

    def _load_agents(self, env):
        """
        Load agents from the agent_paths
        """

        if self.agents:
            return list(self.agents.keys())

        else:
            env_name = env.unwrapped.__class__.__name__
            eval_dir = (
                pathlib.Path(__file__).parent / "evaluation_model_weights" / env_name
            )
            agent_paths = os.listdir(
                eval_dir
            )  # this grabs the available agents for the environment (it is a list of paths to the agents)

            try:
                import stable_baselines3
            except ImportError:
                raise ImportError("Stable Baselines 3 is required to load agents")

            loaded_agents = []
            for agent in agent_paths:
                agent_dir = eval_dir / agent

                model = getattr(stable_baselines3, agent)
                weights = [x for x in agent_dir.iterdir() if x.suffix.lower() == ".zip"]

                if not weights:
                    warnings.warn(f"No weights found for {agent}. Skipping...")
                    continue

                elif len(weights) > 1:
                    warnings.warn(
                        f"Multiple weights found for {agent}. Using the first one."
                    )

                model = model.load(weights[0])

                wrapped_model = ns_gym.base.StableBaselineWrapper(model)

                loaded_agents.append(agent)

                self.agents[agent] = wrapped_model

            return loaded_agents

    def _evaluate_stable_baselines(self, env, agent_list, M):
        """
        Evaluates the baseline_performance of the environment on default environments.
        """

        env_name = env.unwrapped.spec.id

        stationary_env = gym.make(env_name)

        performance = {agent_name: [] for agent_name in agent_list}

        for i, agent_name in enumerate(agent_list):
            agent = self.agents[agent_name]

            for ep in range(M):
                obs, _ = stationary_env.reset()
                done = False
                truncated = False
                total_reward = 0
                while not (done or truncated):
                    action = agent.act(obs)
                    obs, reward, done, truncated, info = stationary_env.step(action)
                    total_reward += reward

                performance[agent_name].append(total_reward)

            performance[agent_name] = sum(performance[agent_name]) / M

        base_ensemble_performance = sum(performance.values()) / len(performance)

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
    r"""Evaluates the difficulty of a transition between two environments using the PAMCTS-Bound metric.

    .. math::
        \forall a \in A: \mid \mid P_t(s'\mid s,a) - P_0(s'\mid a,s)\mid \mid_{\infty}
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, env_1, env_2, verbose=True):
        """
        Evaluate the difficulty of a transition between two environments.

        Args:
            env_1 (gym.Env): The original environment
            env_2 (gym.Env): The new environment
            verbose (bool): Whether to print the results of the evaluation. Defaults to True.

        Returns:
            float: The maximum difference between the transition probabilities of the two environments
        """

        super().evaluate(env_1, env_2)

        if self.space_type == "Box" and self.action_type == "Box":
            raise NotImplementedError

        elif self.space_type == "Discrete" and self.action_type == "Discrete":
            try:
                num_states = env_1.observation_space.n
                num_actions = env_1.action_space.n
                P1 = env_1.unwrapped.P
                P2 = env_2.unwrapped.P
                max_diff = 0
                for s in range(num_states):
                    for a in range(num_actions):
                        for s_prime in range(len(P1[s][a])):
                            assert P1[s][a][s_prime][1] == P2[s][a][s_prime][1], (
                                "Transition probabilities do not match between environments"
                            )
                            max_diff = max(
                                max_diff,
                                abs(P1[s][a][s_prime][0] - P2[s][a][s_prime][0]),
                            )  # From state s with action a, what is the probability of transitioning to state s_prime

                if verbose:
                    self._print_results(max_diff)

                return max_diff

            except Exception as e:
                warnings.warn(f"Error evaluating PAMCTS-Bound: {e}")

        elif self.space_type == "Box" and self.action_type == "Discrete":
            raise NotImplementedError

        elif self.space_type == "Discrete" and self.action_type == "Box":
            raise NotImplementedError

        else:
            raise ValueError("Observation space must be either Box or Discrete")

    def _print_results(self, max_diff):
        print("=" * 40)
        print("Evaluation Results")
        print("=" * 40)
        print(f"PAMCTS-Bound: {max_diff}")
        print("=" * 40)


class BIBO_Stablilty(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, env1, env2):
        """
        Evaluate the stability of the environment.
        """
        raise NotImplementedError


class LyapunovStability(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, env1, env2):
        """
        Evaluate the stability of the environment.
        """
        raise NotImplementedError


class LocalRegret(Evaluator):
    def __init__(
        self, agent, cost_function, learning_rate_eta: float = 0.01, *args, **kwargs
    ) -> None:
        """
        Initializes the LocalRegret evaluator.

        Args:
            agent (AdaptiveAgent): The adaptive agent whose policy is being evaluated.
            cost_function (callable): A differentiable function representing stage cost (h_t).
                                      Should handle cost = -reward.
            learning_rate_eta (float): The eta parameter used in the projected gradient calculation.

        Warning
            This evaluator is still under construction and may not function as intended.
        """
        raise NotImplementedError
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.cost_function = cost_function
        self.eta = learning_rate_eta

        self.historical_trace = []

    def _project_gradient(
        self, grad: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """
        [cite_start]Computes the projected gradient based on Definition 14[cite: 595].
        NOTE: This assumes the constraint set Theta is the entire space.
              For a constrained set, you would need to implement the projection Pi_Theta.
        """

        return grad

    def compute_surrogate_cost(
        self, theta_to_eval: torch.Tensor, t: int, initial_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the surrogate cost F_t(theta) by running a hypothetical simulation
        [cite_start]from time 0 to t using a fixed policy parameter theta_to_eval[cite: 103].
        [cite_start]This simulation uses the true historical dynamics and disturbances[cite: 584].
        """
        hypothetical_state = initial_state.clone()

        for tau in range(t):
            hist_entry = self.historical_trace[tau]

            g_tau, f_tau, a_tau_star = hist_entry["true_dynamics"]

            w_tau = hist_entry["disturbance"]

            hypothetical_u = self.agent.policy(
                hypothetical_state, theta_to_eval, f_tau, a_tau_star
            )

            hypothetical_state = (
                g_tau(hypothetical_state, hypothetical_u, f_tau, a_tau_star) + w_tau
            )

        final_hist_entry = self.historical_trace[t]
        g_t, f_t, a_t_star = final_hist_entry["true_dynamics"]
        final_hypothetical_u = self.agent.policy(
            hypothetical_state, theta_to_eval, f_t, a_t_star
        )

        surrogate_cost_val = self.cost_function(
            hypothetical_state, final_hypothetical_u, theta_to_eval
        )
        return surrogate_cost_val

    def evaluate(self, env: Type[Env], num_steps: int = 1000, *args, **kwargs) -> float:
        """
        Runs an episode for a number of steps and computes the total local regret.

        Args:
            env (Type[Env]): The non-stationary environment, which must be able to provide
                             its true dynamics and disturbances for the evaluation.
            num_steps (int): The total number of time steps (T) to evaluate.

        Returns:
            float: The total local regret R_L(T).
        """
        state, info = env.reset()
        initial_state = torch.tensor(state, dtype=torch.float32)

        # Reset history for the new episode
        self.historical_trace = []
        total_local_regret = 0.0

        # The policy parameter theta evolves over time
        theta_t = self.agent.policy.theta.clone().detach().requires_grad_(True)

        for t in range(num_steps):
            true_dynamics_t = info.get("true_dynamics")  # (g_t, f_t, a_t*)
            disturbance_t = info.get("disturbance")  # w_t
            self.historical_trace.append(
                {"true_dynamics": true_dynamics_t, "disturbance": disturbance_t}
            )

            surrogate_cost = self.compute_surrogate_cost(theta_t, t, initial_state)

            grad_F_t = torch.autograd.grad(surrogate_cost, theta_t, retain_graph=True)[
                0
            ]

            projected_grad = self._project_gradient(grad_F_t, theta_t)

            total_local_regret += torch.sum(projected_grad**2).item()

            action = self.agent.act(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            self.agent.update(state, action, reward, next_state)
            state = next_state

            theta_t = self.agent.policy.theta.clone().detach().requires_grad_(True)

            if terminated or truncated:
                break

        return total_local_regret


if __name__ == "__main__":
    pass
