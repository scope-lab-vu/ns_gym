from typing import Any, SupportsFloat, Type, Union
import gymnasium as gym
import numpy as np
import math
from copy import deepcopy
from ns_gym import base


"""
Wrappers for some of the toytext / gridworld style environments.
"""


class NSCliffWalkingWrapper(base.NSWrapper):
    """Wrapper for gridworld environments that allows for non-stationary transitions.
    
    Args:
        env (gym.Env): The base CliffWalking environment to be wrapped.
        tunable_params (dict[str,base.UpdateFn]]):Dictionary of tunable parameters and their update functions. Currently only supports "P" for transition probabilities.
        change_notification (bool, optional): Do we notify the agent of a change in the MDP.  Defaults to False.
        delta_change_notification (bool, optional): Do we notify the agent of the amount of change in the MDP. Defaults to False.
        initial_prob_dist (list[float], optional): The initial probability distribution over the action space. Defaults to [1,0,0,0].
        modified_rewards (dict[str,int], optional): Set instantanious reward values as such: {"H": -100, "G": 0, "F": -1,"S":-1} where "H" is the hole, "G" is the goal, "F" is the frozen lake and values are the rewards.
        terminal_cliff (bool, optional): Does stepping on the cliff terminate the episode. Defaults to False.

    """

    def __init__(
        self,
        env: Type[gym.Env],
        tunable_params: dict[
            str, Union[Type[base.UpdateFn], Type[base.UpdateDistributionFn]]
        ],
        change_notification: bool = False,
        delta_change_notification: bool = False,
        in_sim_change: bool = False,
        initial_prob_dist=[1, 0, 0, 0],
        modified_rewards: Union[dict[str, int], None] = None,
        terminal_cliff: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            env=env,
            tunable_params=tunable_params,
            change_notification=change_notification,
            delta_change_notification=delta_change_notification,
            in_sim_change=in_sim_change,
            **kwargs,
        )

        self.shape = self.unwrapped.shape
        self.start_state_index = self.unwrapped.start_state_index
        self.nS = self.unwrapped.nS
        self.nA = self.unwrapped.nA

        if modified_rewards:
            self.modified_rewards = modified_rewards
        else:
            self.modified_rewards = {"H": -100, "G": 0, "F": -1, "S": -1}

        self.terminal_cliff = terminal_cliff
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[3, 1:-1] = True

        for state in self.unwrapped.P.keys():
            for action in self.unwrapped.P[state].keys():
                assert len(self.unwrapped.P[state][action]) == 1, (
                    "The base Cliff Walking environment must be non-slippery."
                )

        self.update_fn = tunable_params["P"]
        self.initial_prob_dist = initial_prob_dist
        self.transition_prob = deepcopy(initial_prob_dist)
        self.delta_map_array = np.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8
        )

        # 1. Pre-compute the static outcomes once
        self._outcome_table = self._create_outcome_table()

        # 2. Build the full P-table for the first time
        self.P = self._build_P_from_outcomes(self.transition_prob)
        setattr(self.unwrapped, "P", self.P)
        self.intial_p = self._copy_P(self.P)

    def _create_outcome_table(self):
        """
        Pre-computes the static outcomes (next_state, reward, terminated) for all state-action pairs.
        This is run only once during initialization.
        """
        outcome_table = {}
        for s in range(self.nS):
            outcome_table[s] = {}
            current_pos = np.unravel_index(s, self.shape)
            for a in range(self.nA):
                b_actions = np.array([a, (a + 1) % 4, (a - 1) % 4, (a + 2) % 4])
                deltas = self.delta_map_array[b_actions]
                new_positions = np.array(current_pos) + deltas

                np.clip(
                    new_positions[:, 0], 0, self.shape[0] - 1, out=new_positions[:, 0]
                )
                np.clip(
                    new_positions[:, 1], 0, self.shape[1] - 1, out=new_positions[:, 1]
                )
                new_positions = new_positions.astype(int)

                new_states = np.ravel_multi_index(
                    (new_positions[:, 0], new_positions[:, 1]), self.shape
                )

                is_cliff = self._cliff[new_positions[:, 0], new_positions[:, 1]]
                is_terminated = np.all(
                    new_positions == (self.shape[0] - 1, self.shape[1] - 1), axis=1
                )

                rewards = np.where(
                    is_cliff,
                    self.modified_rewards["H"],
                    np.where(
                        is_terminated,
                        self.modified_rewards["G"],
                        self.modified_rewards["F"],
                    ),
                )
                terminated_flags = np.where(
                    is_cliff, self.terminal_cliff, is_terminated
                )
                next_states = np.where(is_cliff, self.start_state_index, new_states)

                outcome_table[s][a] = list(
                    zip(
                        next_states.tolist(),
                        rewards.tolist(),
                        terminated_flags.tolist(),
                    )
                )
        return outcome_table

    def _build_P_from_outcomes(self, probs):
        """Builds a full P-table by combining probabilities with the pre-computed outcomes."""
        P = {}
        for s in range(self.nS):
            P[s] = {}
            for a in range(self.nA):
                outcomes = self._outcome_table[s][a]
                P[s][a] = [(probs[i],) + outcomes[i] for i in range(4)]
        return P

    def _update_P_probabilities(self):
        """
        Efficiently updates only the probabilities in the existing P-table without rebuilding it.
        This is the new, fast update method.
        """
        new_probs = self.transition_prob
        for s in range(self.nS):
            for a in range(self.nA):
                transitions = self.P[s][a]
                # Overwrite the probability (index 0) in each transition tuple
                self.P[s][a] = [(new_probs[i],) + transitions[i][1:] for i in range(4)]

    def step( self, action: int):
        """
        Args:
            action (int): The action to take in the environment.
    
        Returns:
            tuple[dict, base.Reward, bool, bool, dict[str, Any]]: The observation, reward, termination signal, truncation signal, and additional information.       
        """
        if self.is_sim_env and not self.in_sim_change:
            env_change = {"P": 0}
            delta_change = {"P": 0.0}

            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )
        else:
            self.transition_prob, env_change_flag, delta_change = self.update_fn(
                self.transition_prob, self.t
            )
            if env_change_flag:
                self._update_P_probabilities()

            env_change = {"P": env_change_flag}
            delta_change = {"P": delta_change}

            setattr(self.unwrapped, "P", self.P)
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )

        info["transition_prob"] = self.transition_prob
        return obs, reward, terminated, truncated, info

    # --- The rest of the class remains the same ---
    def _copy_P(self, P):
        return {
            s: {a: transitions[:] for a, transitions in actions.items()}
            for s, actions in P.items()
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.transition_prob = deepcopy(self.initial_prob_dist)
        self.update_fn = self.tunable_params["P"]
        setattr(self.unwrapped, "P", self.intial_p)
        return obs, info

    def get_planning_env(self):
        assert self.has_reset, (
            "The environment must be reset before getting the planning environment."
        )
        if self.is_sim_env or self.change_notification:
            return deepcopy(self)
        elif not self.change_notification:
            planning_env = deepcopy(self)
            planning_env.transition_prob = deepcopy(self.initial_prob_dist)
            setattr(planning_env.unwrapped, "P", self.intial_p)
        return planning_env

    def __deepcopy__(self, memo):
        # env_id = self.unwrapped.spec.id

        env_id = "CliffWalking-v1"
        sim_env = NSCliffWalkingWrapper(
            gym.make(env_id, max_episode_steps=1000),
            tunable_params=deepcopy(self.tunable_params, memo),
            change_notification=self.change_notification,
            delta_change_notification=self.delta_change_notification,
            in_sim_change=self.in_sim_change,
            initial_prob_dist=self.initial_prob_dist,
            scalar_reward=self.scalar_reward,
        )
        memo[id(self)] = sim_env

        sim_env.reset()
        sim_env.unwrapped.s = self.unwrapped.s
        sim_env.t = self.t
        sim_env.transition_prob = self.transition_prob[:]
        sim_env.update_fn = deepcopy(self.update_fn, memo)

        sim_env.intial_p = self._copy_P(self.intial_p)
        current_P = self._copy_P(self.P)
        sim_env.P = current_P
        setattr(sim_env.unwrapped, "P", current_P)

        sim_env.is_sim_env = True
        return sim_env

    def close(self):
        return super().close()

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()


class NSFrozenLakeWrapper(base.NSWrapper):
    """
    A wrapper for the FrozenLake environment that allows for non-stationary transitions.

    Args:
        env (gym.Env): The base FrozenLake environment to be wrapped.
        tunable_params (dict[str,base.UpdateFn]]):Dictionary of tunable parameters and their update functions. Currently only supports "P" for transition probabilities.
        change_notification (bool, optional): Do we notify the agent of a change in the MDP.  Defaults to False.
        delta_change_notification (bool, optional): Do we notify the agent of the amount of change in the MDP. Defaults to False.
        initial_prob_dist (list[float], optional): The initial probability distribution over the action space. Defaults to [1,0,0].
        is_slippery (bool, optional): Is the environment slipperry. . Defaults to True.

    Keyword Args:
        modified_rewards (dict[str,Type[base.UpdateFn]], optional): Set instantanious reward values as such: {"H": -1, "G": 1, "F": 0,"S":0} where "H" is the hole, "G" is the goal, "F" is the frozen lake and values are the rewards.

    """

    def __init__(
        self,
        env: Type[gym.Env],
        tunable_params: dict[
            str, Union[Type[base.UpdateFn], Type[base.UpdateDistributionFn]]
        ],
        change_notification: bool = False,
        delta_change_notification: bool = False,
        in_sim_change: bool = False,
        initial_prob_dist=[1, 0, 0],
        modified_rewards: Union[dict[str, int], None] = None,
        **kwargs: Any,
    ):
        """ """

        super().__init__(
            env=env,
            tunable_params=tunable_params,
            change_notification=change_notification,
            delta_change_notification=delta_change_notification,
            in_sim_change=in_sim_change,
            **kwargs,
        )

        # self.P = getattr(self.unwrapped,"P")

        # for state in self.P.keys():
        #     for action in self.P[state].keys():
        #         assert(len(self.P[state][action]) == 1),("The base FrozenLake environment must be non-slippery,set `env = gym.make(is_slippery = False`.")

        self.modified_rewards = modified_rewards

        self.ncol = self.unwrapped.ncol
        self.nrow = self.unwrapped.nrow
        self.desc = self.unwrapped.desc

        self.nA = self.unwrapped.action_space.n
        self.nS = self.ncol * self.nrow

        self.LEFT = 0
        self.DOWN = 1
        self.RIGHT = 2
        self.UP = 3

        self.delta_t = 1
        self.update_fn = tunable_params["P"]

        assert sum(initial_prob_dist) == 1 or math.isclose(sum(initial_prob_dist), 1), (
            "The sum of transition probabilities must be 1."
        )
        assert len(initial_prob_dist) == 3, (
            "The length of the transition probability distribution must be 3. Each action can have at most 3 possible outcomes."
        )
        self.initial_prob_dist = initial_prob_dist

        self.transition_prob = deepcopy(initial_prob_dist)
        self._update_transition_prob_table()
        setattr(self.unwrapped, "P", self.P)
        self.intial_p = deepcopy(self.P)

    def step(
        self, action: int
    ):
        """
        Args:
            action (int): The action to take in the environment.

        Returns:
            tuple[dict, base.Reward, bool, bool, dict[str, Any]]: The observation, reward, termination signal, truncation signal, and additional information.

        """

        if self.is_sim_env and not self.in_sim_change:
            # action = self._get_action(action)
            env_change = {"P": 0}
            delta_change = {"P": 0.0}
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )
        else:
            self.transition_prob, env_change_flag, delta_change = self.update_fn(
                self.transition_prob, self.t
            )
            if env_change_flag:
                self._update_transition_prob_table()
            setattr(self.unwrapped, "P", self.P)
            assert self.unwrapped.P == self.P, (
                "The transition probability table is not being updated correctly."
            )

            env_change = {"P": env_change_flag}
            delta_change = {"P": delta_change}

            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )
            # obs, reward, terminated, truncated, info = super().step(action,env_change=None,delta_change=None)
        info["transition_prob"] = self.transition_prob
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
            Args:
                seed (int | None, optional): The random seed for initialization. Defaults to None.
                options (dict[str, Any] | None, optional): Additional options for resetting the environment. Defaults to None.

            Returns:
                tuple[Any, dict[str, Any]]: The initial observation and additional information.
        """
        
        obs, info = super().reset(seed=seed, options=options)
        self.transition_prob = deepcopy(self.initial_prob_dist)
        self.update_fn = self.tunable_params["P"]
        setattr(self.unwrapped, "P", self.intial_p)
        return obs, info

    def state_encoding(self, row, col):
        return row * self.ncol + col

    def state_decoding(self, state):
        return state // self.ncol, state % self.ncol

    def close(self):
        return super().close()

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()

    def _get_action(self, action: int):
        # prob_n = np.asarray(self.transition_prob)
        # csprob_n = np.cumsum(prob_n)
        # a = np.argmax(csprob_n > np.random.random())
        # inds = [0,1,-1]
        ind = np.random.choice([0, 1, -1], p=self.transition_prob)
        # ind = inds[a]
        action = (action + ind) % 4
        return action

    def _update_transition_prob_table(self):
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self.to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        for ind, b in enumerate([a, (a + 1) % 4, (a - 1) % 4]):
                            li.append(
                                (
                                    self.transition_prob[ind],
                                    *self.update_probability_matrix(row, col, b),
                                )
                            )

    def to_s(self, row, col):
        return row * self.ncol + col

    def inc(self, row, col, a):
        if a == self.LEFT:
            col = max(col - 1, 0)
        elif a == self.DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == self.RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == self.UP:
            row = max(row - 1, 0)
        return (row, col)

    def update_probability_matrix(self, row, col, action):
        newrow, newcol = self.inc(row, col, action)
        newstate = self.to_s(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        terminated = bytes(newletter) in b"GH"
        if self.modified_rewards:
            reward = float(self.modified_rewards[newletter.decode("utf-8")])
        else:
            reward = float(newletter == b"G")
        return newstate, reward, terminated

    def get_planning_env(self):
        assert self.has_reset, (
            "The environment must be reset before getting the planning environment."
        )
        if self.is_sim_env or self.change_notification:
            return deepcopy(self)
        elif not self.change_notification:
            planning_env = deepcopy(self)
            planning_env.transition_prob = deepcopy(self.initial_prob_dist)
            setattr(planning_env.unwrapped, "P", self.intial_p)
        return planning_env

    def __deepcopy__(self, memo):
        # env_id = self.unwrapped.spec.id
        env_id = "FrozenLake-v1"

        sim_env = gym.make(
            env_id, is_slippery=False, render_mode="ansi", 
        )
        sim_env = NSFrozenLakeWrapper(
            sim_env,
            tunable_params=deepcopy(self.tunable_params),
            change_notification=self.change_notification,
            delta_change_notification=self.delta_change_notification,
            in_sim_change=self.in_sim_change,
            initial_prob_dist=self.initial_prob_dist,
            modified_rewards=self.modified_rewards,
            scalar_reward=self.scalar_reward,
        )
        sim_env.reset()
        sim_env.unwrapped.s = deepcopy(self.unwrapped.s)
        # sim_env._elapsed_steps = self._elapsed_steps
        sim_env.t = deepcopy(self.t)
        sim_env.transition_prob = deepcopy(self.transition_prob)
        sim_env.update_fn = deepcopy(self.update_fn)
        sim_env.intial_p = deepcopy(self.intial_p)
        sim_env.unwrapped.P = deepcopy(self.P)
        sim_env.is_sim_env = True
        return sim_env


class NSBridgeWrapper(base.NSWrapper):
    """Bridge environment wrapper that allows for non-stationary transitions."""

    def __init__(
        self,
        env: Type[gym.Env],
        tunable_params: dict[
            str, Union[Type[base.UpdateFn], Type[base.UpdateDistributionFn]]
        ],
        change_notification: bool = False,
        delta_change_notification: bool = False,
        in_sim_change: bool = False,
        initial_prob_dist=[1, 0, 0],
        modified_rewards: Union[dict[str, int], None] = None,
        **kwargs: Any,
    ):
        super().__init__(
            env=env,
            tunable_params=tunable_params,
            change_notification=change_notification,
            delta_change_notification=delta_change_notification,
            in_sim_change=in_sim_change,
            **kwargs,
        )

        self.initial_prob_dist = initial_prob_dist
        setattr(self.unwrapped, "P", initial_prob_dist)
        assert self.unwrapped.P == initial_prob_dist, (
            "The initial probability distribution is not being set correctly."
        )
        self.tunable_params = tunable_params
        self.init_tunable_params = deepcopy(tunable_params)
        self.update_fn = tunable_params["P"]

        self.delta_t = 1

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.is_sim_env and not self.in_sim_change:
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=None, delta_change=None
            )
        else:
            P = self.unwrapped.P
            P, env_change, delta_change = self.update_fn(P, self.t)
            setattr(self.unwrapped, "P", P)
            assert self.unwrapped.P == P, (
                "The transition probability table is not being updated correctly."
            )
            obs, reward, terminated, truncated, info = super().step(
                action, env_change=env_change, delta_change=delta_change
            )
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.unwrapped.P = deepcopy(self.initial_prob_dist)
        self.tunable_params = deepcopy(self.init_tunable_params)
        self.update_fn = self.tunable_params["P"]
        return obs, info

    def get_planning_env(self):
        assert self.has_reset, (
            "The environment must be reset before getting the planning environment."
        )
        if self.is_sim_env or self.change_notification:
            return deepcopy(self)
        elif not self.change_notification:
            planning_env = deepcopy(self)
            setattr(planning_env.unwrapped, "P", deepcopy(self.initial_prob_dist))
        return planning_env

    def __deepcopy__(self, memo):
        sim_env = gym.make("ns_bench/Bridge-v0", max_episode_steps=1000)
        sim_env = NSBridgeWrapper(
            sim_env,
            tunable_params=deepcopy(self.tunable_params),
            change_notification=self.change_notification,
            delta_change_notification=self.delta_change_notification,
            in_sim_change=self.in_sim_change,
            initial_prob_dist=self.initial_prob_dist,
            scalar_reward=self.scalar_reward,
        )
        sim_env.reset()
        sim_env.unwrapped.s = deepcopy(self.unwrapped.s)
        sim_env.t = deepcopy(self.t)
        sim_env.unwrapped.P = deepcopy(self.unwrapped.P)
        sim_env.update_fn = deepcopy(self.update_fn)
        sim_env.is_sim_env = True
        return sim_env

    @property
    def transition_matrix(self):
        return self.unwrapped.transition_matrix


if __name__ == "__main__":
    import gymnasium as gym
    import ns_gym

    env = gym.make("ns_bench/Bridge-v0")
    scheduler = ns_gym.schedulers.ContinuousScheduler()
    update_function = ns_gym.update_functions.DistributionDecrmentUpdate(
        scheduler, k=0.1
    )
    tunable_params = {"P": update_function}

    env = ns_gym.wrappers.NSBridgeWrapper(
        env,
        tunable_params=tunable_params,
        initial_prob_dist=[1, 0, 0],
        change_notification=False,
        delta_change_notification=False,
        in_sim_change=False,
    )

    reward_list = []

    obs, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    while not done and not truncated:
        planning_env = env.get_planning_env()
        agent = ns_gym.benchmark_algorithms.MCTS(
            planning_env, obs, d=200, m=100, c=1.44, gamma=0.99
        )
        action, _ = agent.search()
        print("real env P", env.unwrapped.P)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        episode_reward += reward.reward

    reward_list.append(episode_reward)

    print(np.mean(reward_list))
