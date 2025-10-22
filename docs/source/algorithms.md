# NS-Gym Algorithms


```{eval-rst}
NS-Gym has several built-in benchmark algorithms for training and evaluating agents in non-stationary environments. These algorithms are implemented in the :py:mod:`ns_gym.benchmark_algorithms <ns_gym.benchmark_algorithms>` module. 


.. attention::
   :name: leaderboard

   Detailed descriptions of included algorithms pages are currently under development. For now please take a look at the API documentation for :py:mod:`ns_gym.benchmark_algorithms <ns_gym.benchmark_algorithms>`.


NS-Gym has implementations of the following algorithms:

- :py:class:`Monte Carlo Tree Search <ns_gym.benchmark_algorithms.MCTS>`
- :py:class:`Proximal Policy Optimization <ns_gym.benchmark_algorithms.PPO>`
- :py:class:`Deep Q-Network <ns_gym.benchmark_algorithms.DQN>`
- :py:class:`AlphaZero <ns_gym.benchmark_algorithms.AlphaZeroAgent>`
- :py:class:`Policy Augmented MCTS <ns_gym.benchmark_algorithms.PAMCTS>`
- `ADA-MCTS` (Code included and runnable on NS-Gym envs though needs to be refactored for API compliance)
- `RATS` (Code included and runnable on NS-Gym envs but needs to be refactored for API compliance)

We also support integration with `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_, allowing users to leverage a wide range of RL algorithms in NS-Gym environments.



