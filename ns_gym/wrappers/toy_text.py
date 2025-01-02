from typing import Any, SupportsFloat, Type, Union
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import math
from gymnasium import Wrapper
from copy import deepcopy
from ns_gym import base, update_functions, schedulers

from ns_gym.benchmark_algorithms import algo_utils
from ns_gym.utils import update_probability_table,state_action_update,n_choose_k




"""
Wrappers for some of the toytext / gridworld style environments.
"""

class NSCliffWalkingWrapper(base.NSWrapper):
    """Wrapper for gridworld environments (FrozenLake, CliffWalking, Bridge, ... ) that allows for non-stationary transitions.

    #TODO: This implementation is not finished.

    # NOTE: This may just be a wrapper for the CliffWalking environment. The reason is that although the action spaces for the gridworld environments are
    the same, at each time step the available actions are different.

    This wrapper as opossed to the NSFrozenLakeWrapper simply changes the actions the agent can take given an intended action.
    
    The action space for these environements is discrete. 
    In frozen lake there is probality of going in the intended direction and a probability of going in the two perpendicular directions.
    In cliff walking there is a probality of going in the intended direction and a probability of going in the two perpendicular directions and the reverse direcition. 

    """
    def __init__(self,
                 env: Type[gym.Env], 
                 tunable_params: dict[str,Union[Type[base.UpdateFn], Type[base.UpdateDistributionFn]]], 
                 change_notification: bool = False, 
                 delta_change_notification: bool = False, 
                 in_sim_change: bool = False,
                 initial_prob_dist = [1,0,0,0],
                 modified_rewards: Union[dict[str,int],None] = None,
                 terminal_cliff: bool = False,
                 **kwargs: Any):
        """
        Args:
            env (gym.Env): _description_
            tunable_params (dict[str,Type[base.UpdateDistributionFn]]): A dictionary of parameter names and their associated update functions.
            change_notification` (bool, optional): Flag to indicate whether to notify the agent of changes in the environment. Defaults to False.
            delta_change_notification` (bool, optional): Flag to indicate whether to notify the agent of the amount of change in the transition function. Defaults to False.
            initial_prob_dist (list[float], optional): The initial probability distribution over the action space. Defaults to [1,0,0,0]. 

        Notes:
            We initilize the environment to include and transition probability distribution over the action space called `transition_prob`.
            Each element of the list is the probability of taking an action. The 0th element is the probability of taking the intended action.
            The 1st and 2nd elements are the probabilities of taking the perpendicular actions. The 3rd element is the probability of taking the reverse action.
            The transitions are encoded such that transition_prob[(action - i)%4] is the probabililty of taking an action. i is the index of the list 
        """
   
        super().__init__(env = env,
                         tunable_params = tunable_params,
                         change_notification = change_notification,
                         delta_change_notification = delta_change_notification,
                         in_sim_change = in_sim_change)
        
              
        self.P = self.unwrapped.P
        self.shape = self.unwrapped.shape
        self.start_state_index = self.unwrapped.start_state_index
        self.nS = self.unwrapped.nS
        self.nA = self.unwrapped.nA

        if modified_rewards:
            self.modified_rewards = modified_rewards
        else:
            self.modified_rewards = {"H": -100, "G": 0, "F": -1,"S": -1} #hole , goal , frozen lake, start state 
        
        self.terminal_cliff = terminal_cliff

        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[3, 1:-1] = True

        for state in self.P.keys():
            for action in self.P[state].keys():
                assert(len(self.P[state][action]) == 1),("The base Cliff Walking environment must be non-slippery,set `env = gym.make(is_slippery = False`.")
        
        self.delta_t = 1
        self.update_fn = tunable_params["P"]
        self.action_space = self.unwrapped.action_space
        
        #The transitions are encoded such that transition_prob[(action - i)%4] is the probabililty of taking an action. i is the index of the list 
        self.initial_prob_dist = initial_prob_dist
        self.transition_prob  =  deepcopy(initial_prob_dist) #This is the initial probability distribution over the action space.


        self.update_transition_prob_table()
        setattr(self.unwrapped,"P",self.P)
        self.intial_p = deepcopy(self.P)

    def step(self,action: int) -> tuple[Type[base.Observation], Type[base.Reward], bool, bool, dict[str, Any]]:
        """
        Args:
            action (int): The action to take in the environment.
        
        Returns:
            tuple[base.Observation, base.Reward, bool, bool, dict[str, Any]]: The observation, reward, termination signal, truncation signal, and additional information.

        """

        if self.is_sim_env and not self.in_sim_change:
            # action = self._get_action(action)
            obs,reward,terminated,truncated,info = super().step(action,env_change=None,delta_change=None)
        else:
            self.transition_prob, env_change, delta_change = self.update_fn(self.transition_prob,self.t) 
            self.update_transition_prob_table()   
            setattr(self.unwrapped,"P",self.P)
            assert(self.unwrapped.P == self.P),("The transition probability table is not being updated correctly.")
            obs, reward, terminated, truncated, info = super().step(action,env_change=env_change,delta_change=delta_change)
            # obs, reward, terminated, truncated, info = super().step(action,env_change=None,delta_change=None)
        info["transition_prob"] = self.transition_prob
        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Args:
            seed (int | None, optional): _description_. Defaults to None.
            options (dict[str, Any] | None, optional): _description_. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: _description_
        """
        obs,info = super().reset(seed=seed, options=options)
        self.transition_prob = deepcopy(self.initial_prob_dist)
        self.update_fn = self.tunable_params["P"]
        setattr(self.unwrapped,"P",self.intial_p)
        return obs, info
    
    def close(self):
        return super().close()
    
    def __str__(self):
        return super().__str__()
    
    def __repr__(self):
        return super().__repr__()
    
    def update_transition_prob_table(self):
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self.calculate_transition_prob(position, UP)
            self.P[s][RIGHT] = self.calculate_transition_prob(position, RIGHT)
            self.P[s][DOWN] = self.calculate_transition_prob(position, DOWN)
            self.P[s][LEFT] = self.calculate_transition_prob(position, LEFT)

    def new_limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def calculate_transition_prob(self, current,action):
        """Determine the outcome for an action. Transition Prob is always 1.0.

        Args:
            current: Current position on the grid as (row, col)
            action: Action to be taken as int. 0: up, 1: right, 2: down, 3: left

        Returns:
            Tuple of ``(1.0, new_state, reward, terminated)``
        """

        delta_map = {0: [-1,0], 1: [0,1], 2: [1,0], 3: [0,-1]}

        transition_list = []

        for ind,b in enumerate([action, (action + 1)%4, (action - 1)%4,(action + 2)%4]):
            new_position = np.array(current) + np.array(delta_map[b])
            new_position = self.new_limit_coordinates(new_position).astype(int)
            new_state = np.ravel_multi_index(tuple(new_position), self.shape)

            terminal_state = (self.shape[0] - 1, self.shape[1] - 1) # corner of the map 

            is_terminated = tuple(new_position) == terminal_state

            if self._cliff[tuple(new_position)]:
                transition_list.append((self.transition_prob[ind], self.start_state_index, self.modified_rewards["H"], self.terminal_cliff))
            elif is_terminated:
                transition_list.append((self.transition_prob[ind], new_state, self.modified_rewards["G"], is_terminated))
            else:
                transition_list.append((self.transition_prob[ind], new_state, self.modified_rewards["F"], is_terminated))

        return transition_list

    
    def get_planning_env(self):
        assert(self.has_reset),("The environment must be reset before getting the planning environment.")
        if self.is_sim_env or self.change_notification:
            return deepcopy(self)
        elif not self.change_notification:
            planning_env = deepcopy(self)
            planning_env.transition_prob = deepcopy(self.initial_prob_dist)
            setattr(planning_env.unwrapped,"P",self.intial_p)
        return planning_env
    
    def __deepcopy__(self, memo):
        sim_env = gym.make("CliffWalking-v0",max_episode_steps=1000)
        sim_env = NSCliffWalkingWrapper(sim_env,
                                      tunable_params=deepcopy(self.tunable_params),
                                      change_notification=self.change_notification,
                                      delta_change_notification=self.delta_change_notification,
                                      in_sim_change=self.in_sim_change,
                                      initial_prob_dist=self.initial_prob_dist)
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
    

class NSFrozenLakeWrapper(base.NSWrapper):
    """
    A wrapper for the FrozenLake environment that allows for non-stationary transitions.
    """
    def __init__(self, 
                 env: Type[gym.Env], 
                 tunable_params: dict[str,Union[Type[base.UpdateFn], Type[base.UpdateDistributionFn]]], 
                 change_notification: bool = False, 
                 delta_change_notification: bool = False, 
                 in_sim_change: bool = False,
                 initial_prob_dist = [1,0,0],
                 modified_rewards: Union[dict[str,int],None] = None,
                 **kwargs: Any):
        """

        Args:
            env (gym.Env): _description_
            tunable_params (dict[str,Type[base.NSTransitionFn]]): _description_
            change_notification (bool, optional): Do we notify the agent of a change in the MDP.  Defaults to False.
            delta_change_notification (bool, optional): Do we notify the agent of the amount of change in the MDP. Defaults to False.
            initial_prob_dist (list[float], optional): The initial probability distribution over the action space. Defaults to [1,0,0].
            is_slippery (bool, optional): Is the environment slipperry. . Defaults to True.

        Keywork Args:
            modified_rewards (dict[str,Type[base.UpdateFn]], optional): Set instantanious reward values as such: {"H": -1, "G": 1, "F": 0,"S":0} where "H" is the hole, "G" is the goal, "F" is the frozen lake and values are the rewards.

        """
        

        super().__init__(env = env,
                         tunable_params = tunable_params,
                         change_notification = change_notification,
                         delta_change_notification = delta_change_notification,
                         in_sim_change = in_sim_change,
                         **kwargs)
        
        # self.P = getattr(self.unwrapped,"P")

        # for state in self.P.keys():
        #     for action in self.P[state].keys():
        #         assert(len(self.P[state][action]) == 1),("The base FrozenLake environment must be non-slippery,set `env = gym.make(is_slippery = False`.")

        self.modified_rewards = modified_rewards

        self.ncol = self.unwrapped.ncol
        self.nrow = self.unwrapped.nrow
        self.desc = self.unwrapped.desc

        self.nA = self.unwrapped.action_space.n
        self.nS = self.ncol*self.nrow
        
        self.LEFT = 0
        self.DOWN = 1 
        self.RIGHT = 2
        self.UP = 3

        self.delta_t = 1
        self.update_fn = tunable_params["P"]
        
        assert(sum(initial_prob_dist) == 1 or math.isclose(sum(initial_prob_dist),1)),("The sum of transition probabilities must be 1.")
        assert(len(initial_prob_dist) == 3),("The length of the transition probability distribution must be 3. Each action can have at most 3 possible outcomes.") 
        self.initial_prob_dist = initial_prob_dist

        self.transition_prob = deepcopy(initial_prob_dist)
        self._update_transition_prob_table()
        setattr(self.unwrapped,"P",self.P)
        self.intial_p = deepcopy(self.P)


    def step(self,action: int) -> tuple[Type[base.Observation], Type[base.Reward], bool, bool, dict[str, Any]]:
        """
        Args:
            action (int): The action to take in the environment.
        
        Returns:
            tuple[base.Observation, base.Reward, bool, bool, dict[str, Any]]: The observation, reward, termination signal, truncation signal, and additional information.

        """

        if self.is_sim_env and not self.in_sim_change:
            # action = self._get_action(action)
            obs,reward,terminated,truncated,info = super().step(action,env_change=None,delta_change=None)
        else:
            self.transition_prob, env_change, delta_change = self.update_fn(self.transition_prob,self.t) 
            self._update_transition_prob_table()   
            setattr(self.unwrapped,"P",self.P)
            assert(self.unwrapped.P == self.P),("The transition probability table is not being updated correctly.")
            obs, reward, terminated, truncated, info = super().step(action,env_change=env_change,delta_change=delta_change)
            # obs, reward, terminated, truncated, info = super().step(action,env_change=None,delta_change=None)
        info["transition_prob"] = self.transition_prob
        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Args:
            seed (int | None, optional): _description_. Defaults to None.
            options (dict[str, Any] | None, optional): _description_. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: _description_
        """
        obs,info = super().reset(seed=seed, options=options)
        self.transition_prob = deepcopy(self.initial_prob_dist)
        self.update_fn = self.tunable_params["P"]
        setattr(self.unwrapped,"P",self.intial_p)
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
    
    def _get_action(self,action: int):
        # prob_n = np.asarray(self.transition_prob)
        # csprob_n = np.cumsum(prob_n)
        # a = np.argmax(csprob_n > np.random.random())
        # inds = [0,1,-1]
        ind = np.random.choice([0,1,-1],p=self.transition_prob)
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
                        for ind,b in enumerate([a, (a + 1)%4, (a - 1)%4]):
                            li.append(
                                (self.transition_prob[ind], *self.update_probability_matrix(row, col, b))
                            )
        

        
    def to_s(self,row, col):
        return row * self.ncol + col

    def inc(self,row, col, a):
        if a == self.LEFT:
            col = max(col - 1, 0)
        elif a == self.DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == self.RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == self.UP:
            row = max(row - 1, 0)
        return (row, col)

    def update_probability_matrix(self,row, col, action):
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

        assert(self.has_reset),("The environment must be reset before getting the planning environment.")
        if self.is_sim_env or self.change_notification:
            return deepcopy(self)
        elif not self.change_notification:
            planning_env = deepcopy(self)
            planning_env.transition_prob = deepcopy(self.initial_prob_dist)
            setattr(planning_env.unwrapped,"P",self.intial_p)
        return planning_env
    

    def __deepcopy__(self, memo):
        #TODO: You dont need all these deepcopy calls.
        sim_env = gym.make("FrozenLake-v1",is_slippery=False,max_episode_steps=1000,render_mode="ansi")
        sim_env = NSFrozenLakeWrapper(sim_env,
                                      tunable_params=deepcopy(self.tunable_params),
                                      change_notification=self.change_notification,
                                      delta_change_notification=self.delta_change_notification,
                                      in_sim_change=self.in_sim_change,
                                      initial_prob_dist=self.initial_prob_dist,
                                      modified_rewards=self.modified_rewards)
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
    
        
    # def __deepcopy__(self, memo):
    #     f = self.__deepcopy__
    #     self.__deepcopy__ = None
    #     sim_env = deepcopy(self)
    #     sim_env.__deepcopy__ = f
    #     sim_env.is_sim_env = True   
    #     return sim_env




class NSTaxiWrapper(Wrapper):
    """A wrapper for the Taxi gridworld environment. 
    TODO: This is a prototype and needs to be fixed.
    Args:
        Wrapper (_type_): _description_

    Notes:
        I think we can make the Taxi environment non-stationary in a couple of ways.
        
        1. We can add changing pick up/movement probabilites
        2. We can create non-stationary rewards by dynamically adding passengers to be picked up.
            At each time step there is some probability that new passengers arrive. 

        The ussual appraoch of specifying the full MDP may eb 
    ## Starting State
    
    The episode starts with the player in a random state.

    ## Rewards
    
    The agent rewards are largely the same. 

    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10  executing "pickup" and "drop-off" actions illegally.

    ## Termination conditions

    The episode finishes when the max time limit is hit or when a passenger is correclty dropped off. 

    ## Non-stationary passengers. 

    There can be multiple passengers with reward values that vary ofer time. 
    The reward of a passenger must be some positive number between 0 and 20. The reward changes wrt time until the passenger is picked up.
    After the taxi picks up the passenger the reward stays the same until drop off. 
    After dorp off the episode ends? 

    """

    def __init__(self, env: gym.Env):
        raise NotImplementedError("This method is not implemented.")


class NSBridgeWrapper(base.NSWrapper):
    """Bridge environment wrapper that allows for non-stationary transitions.
    """
    def __init__(self, 
                 env: Type[gym.Env], 
                 tunable_params: dict[str,Union[Type[base.UpdateFn], Type[base.UpdateDistributionFn]]], 
                 change_notification: bool = False, 
                 delta_change_notification: bool = False, 
                 in_sim_change: bool = False,
                 initial_prob_dist = [1,0,0],
                 modified_rewards: Union[dict[str,int],None] = None):
        
        super().__init__(env = env,
                    tunable_params = tunable_params,
                    change_notification = change_notification,
                    delta_change_notification = delta_change_notification,
                    in_sim_change = in_sim_change)
        
        self.initial_prob_dist = initial_prob_dist
        setattr(self.unwrapped,"P",initial_prob_dist)
        assert(self.unwrapped.P == initial_prob_dist),("The initial probability distribution is not being set correctly.")
        self.tunable_params = tunable_params
        self.init_tunable_params = deepcopy(tunable_params)
        self.update_fn = tunable_params["P"]

        
        # self.P = getattr(self.unwrapped,"P")
        # assert(self.P == initial_prob_dist),("The initial probability distribution is not being set correctly.")
        # self.intial_p = deepcopy(self.unwrapped.P)
        self.delta_t = 1

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.is_sim_env and not self.in_sim_change:
            # action = self._get_action(action)
            obs,reward,terminated,truncated,info = super().step(action,env_change=None,delta_change=None)
        else:
            P = self.unwrapped.P
            P, env_change, delta_change = self.update_fn(P,self.t) 
            # print(P)
            setattr(self.unwrapped,"P",P)
            # self.P = P
            assert(self.unwrapped.P == P),("The transition probability table is not being updated correctly.")
            obs, reward, terminated, truncated, info = super().step(action,env_change=env_change,delta_change=delta_change)
            # obs, reward, terminated, truncated, info = super().step(action,env_change=None,delta_change=None)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs,info = super().reset(seed=seed, options=options)
        self.unwrapped.P = deepcopy(self.initial_prob_dist)
        self.tunable_params = deepcopy(self.init_tunable_params)
        self.update_fn = self.tunable_params["P"]
        # setattr(self.unwrapped,"P",self.initial_prob_dist)
        return obs, info

    def get_planning_env(self):

        assert(self.has_reset),("The environment must be reset before getting the planning environment.")
        if self.is_sim_env or self.change_notification:
            return deepcopy(self)
        elif not self.change_notification:
            planning_env = deepcopy(self)
            # planning_env.transition_prob = deepcopy(self.initial_prob_dist)
            setattr(planning_env.unwrapped,"P",deepcopy(self.initial_prob_dist))
        return planning_env
    

    def __deepcopy__(self, memo):
        #TODO: You dont need all these deepcopy calls.
        sim_env = gym.make("ns_bench/Bridge-v0",max_episode_steps=1000)
        sim_env = NSBridgeWrapper(sim_env,tunable_params=deepcopy(self.tunable_params),change_notification=self.change_notification,delta_change_notification=self.delta_change_notification,in_sim_change=self.in_sim_change,initial_prob_dist=self.initial_prob_dist)
        sim_env.reset()
        sim_env.unwrapped.s = deepcopy(self.unwrapped.s)
        # sim_env._elapsed_steps = self._elapsed_steps
        sim_env.t = deepcopy(self.t)
        sim_env.unwrapped.P = deepcopy(self.unwrapped.P)
        sim_env.update_fn = deepcopy(self.update_fn)
        # sim_env.intial_p = deepcopy(self.intial_p)
        # sim_env.unwrapped.P = deepcopy(self.P)
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
    # update_function = ns_bench.update_functions.DistributionNoUpdate(scheduler)
    update_function = ns_gym.update_functions.DistributionDecrmentUpdate(scheduler,k=0.1)
    tunable_params = {"P": update_function} 

    env = ns_gym.wrappers.NSBridgeWrapper(env,
                                            tunable_params=tunable_params,
                                            initial_prob_dist=[1,0,0],
                                            change_notification=False,
                                            delta_change_notification=False,
                                            in_sim_change=False)    
    
    reward_list= []

    obs,_ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    while not done and not truncated:
        planning_env = env.get_planning_env()
        agent = ns_gym.benchmark_algorithms.MCTS(planning_env,obs,d=200,m=100,c=1.44,gamma=0.99)
        action,_ = agent.search()
        print("real env P",env.unwrapped.P)
        obs,reward,done,truncated,info = env.step(action)
        env.render()
        episode_reward += reward.reward

    reward_list.append(episode_reward)

    print(np.mean(reward_list))







