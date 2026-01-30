import gymnasium as gym
import numpy as np
from copy import deepcopy
import ns_gym as nsg

import ns_gym.base as base
import random


"""
MCTS with Chance Nodes to handle stochastic environments. This implementation used a global table to store the Q values and visit counts for state-action pairs and states. Compatible with OpenAI Gym environments.
"""

class DecisionNode:
    """
    Decision node class, labelled by a state.
    """
    def __init__(self, parent, state, weight, is_terminal,reward):
        """
        Args:
            parent (ChanceNode): The parent node of the decision node.
            state (Union[int,np.ndarray]): Environment state.
            weight (float): Probability to occur given the parent (state-action pair)
            is_terminal (bool): Is the state terminal.
            reward (float): immediate reward for reaching this state.

        Attributes:
            children (list): List of child nodes.
            value (float): Value of the state.
            weighted_value (float): Weighted value of the state.

        """
        self.parent = parent
        state, _ = nsg.utils.type_mismatch_checker(observation=state,reward=None)

        assert not isinstance(state, dict), "State is still a dict after type checking."
   

        if isinstance(state, dict) and 'state' in state:
            state = state['state']
        if isinstance(state, np.ndarray):
            state = tuple(state)


        self.state = state
        self.weight = weight  # Probability to occur
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.value = 0 # value of state
        self.reward = reward# immediate reward
        self.weighted_value = self.weight * self.value

class ChanceNode:
    """
    Chance node class, labelled by a state-action pair.
    The state is accessed via the parent attribute.
    """
    def __init__(self, parent, action):
        """
        Args:
            parent (DecicionsNode): Parent node of the chance node, a decision node.
            action (int): Action taken from the parent node, ie state_1 has child (state_2,action_1) say 
        
        Attributes:
            children (list): List of child nodes (DecisionNode)
            value (float): Value of the state-action pair.
            depth (int): Depth of the node in the tree.
        """
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []
        self.value = 0

class MCTS(base.Agent):
    """Vanilla MCTS with Chance Nodes. Compatible with OpenAI Gym environments.
        Selection and expansion are combined into the "treepolicy method"
        The rollout/simulation is the "default" policy.

    Args:
        env (gym.Env): The environment to run the MCTS on.
        state (Union[int, np.ndarray]): The state to start the MCTS from.
        d (int): The depth of the MCTS.
        m (int): The number of simulations to run.
        c (float): The exploration constant.
        gamma (float): The discount factor.

    Attributes:
        v0 (DecisionNode): The root node of the tree.
        possible_actions (list): List of possible actions in the environment.
        Qsa (dict): Dictionary to store Q values for state-action pairs.
        Nsa (dict): Dictionary to store visit counts for state-action pairs.
        Ns (dict): Dictionary to store visit counts for states.
    """
    def __init__(self,env:gym.Env,state,d,m,c,gamma) -> None:
        """
       

        """
        self.env = env # This is the current state of the mdp
        self.d = d # depth 
        self.m = m # number of simulations
        self.c = c # exploration constant


        state, _ = nsg.utils.type_mismatch_checker(observation=state,reward=None)
        self.v0 = DecisionNode(parent=None,state=state,weight=1,is_terminal=False,reward=0)

        if not isinstance(env.action_space,gym.spaces.Discrete):
            raise ValueError("Only discrete action spaces are supported")
        
        self.possible_actions = [x for x in range(env.action_space.n)]
        self.gamma = gamma        
        self.Qsa = {}  # stores Q values for s,a pairs, defaults to Qsa of 0
        self.Nsa = {}  # stores visit counts for s,a pairs, default to Nsa of 0
        self.Ns = {} # stores visit counts for states, default to Ns of 0

    def search(self):
        """Do the MCTS by doing m simulations from the current state s. 
        After doing m simulations we simply choose the action that maximizes the estimate of Q(s,a)

        Returns:
            best_action(int): best action to take
            action_values(list): list of Q values for each action.
        """
        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deep copy of of the og env at the root nod 
            vl = self._tree_policy(self.v0) #vl is the last node visitied by the tree search as chance node
            expanded_node = self._expand(vl) 
            if type(expanded_node) == ChanceNode:
                expanded_node = self._expand(expanded_node) #DecisionNode
            R = self._default_policy(expanded_node) #R is the reward from the simulation (default policy)
            self._backpropagation(R,expanded_node)


        ba = self.best_action(self.v0) # best action
        action_values = [self.Qsa[(self.v0.state,a)] for a in self.possible_actions] # Q values for s a pairs
        ba = np.argmax(action_values)
        return ba,action_values

    def _tree_policy(self, node) -> ChanceNode:
        """Tree policy for MCTS. Traverse the tree from the root node to a leaf node.
        Args:
            node (DecisionNode): The root node of the tree.
        Returns:
            ChanceNode: The leaf node reached by the tree policy.
        """
        while node.children:
            if type(node) == DecisionNode:
                node = self._selection(node)
                assert(type(node) == ChanceNode)
            else: # chance node
                assert(type(node) == ChanceNode)
                node = self._expand(node) 
                assert(type(node) == DecisionNode),f"got {type(node)} instead of DecisionNode"
        return node

    def _default_policy(self,v:DecisionNode):
        """Simulate/Playout step 
        While state is non-terminal choose  an action uniformly at random, transition to new state. Return the reward for final  state. 

        Args:
            v (DecisionNode): The node to start the simulation from.
        """
        if v.is_terminal:
            return v.reward
        tot_reward = 0
        terminated = False
        truncated = False
        depth = 0
        while not terminated and depth < self.d and not truncated:
            action = np.random.choice(self.possible_actions)
            observation,reward,terminated,truncated,info = self.sim_env.step(action)
            observation ,reward = self.type_checker(observation,reward)
            tot_reward += reward*self.gamma**depth
            depth+=1
        return tot_reward

    def _selection(self,v:DecisionNode):
        """Pick the next node to go down in the search tree based on UTC formula.
        """
        best_child = self.best_child(v)
        return best_child

    def _expand(self,node):
        """Expand the tree by adding a new node to the tree. Handles both decision and chance nodes.
        """
        
        if type(node) == DecisionNode:
            if node.is_terminal:
                return node
            for a in range(self.sim_env.action_space.n):
                new_node = ChanceNode(parent=node,action=a)
                node.children.append(new_node)
            return np.random.choice(node.children) 

        else: # chance node
            action = node.action
            assert(type(node)==ChanceNode)
            obs,reward,term,_,info = self.sim_env.step(action)
            obs,reward = self.type_checker(obs,reward)
            existing_child = [child for child in node.children if child.state == obs]
            if existing_child:
                return existing_child[0]
            else:
                if "prob" in info:
                    w = info["prob"]
                else:
                    w = 1    
                new_node = DecisionNode(parent=node,state=obs,weight=w,is_terminal=term,reward=reward)
                node.children.append(new_node)
                return new_node

    def _backpropagation(self,R,v,depth=0):
        """Backtrack to update the number of times a node has beenm visited and the value of a node untill we reach the root node. 
        """
        depth = 0 
        while v:
            v.value += R
            if type(v) == ChanceNode:
                assert not isinstance(v.parent.state, dict), "Parent state is still a dict after type checking."
                self.update_metrics_chance_node(v.parent.state,v.action,R)
            else:
                self.update_metrics_decision_node(v.state)
            R = R*(self.gamma**depth)
            depth+=1
            v = v.parent

    def update_metrics_chance_node(self, state, action, reward):
        """Update the Q values and visit counts for state-action pairs and states.

        Args:
            state (Union[int,np.ndarray]): The state.
            action (Union[int,float,np.ndarray]): action taken at the state.
            reward (float): The reward received after taking the action at the state.
        """

        if isinstance(state, np.ndarray):
            state = tuple(state)

        if isinstance(action, np.ndarray):
            action = tuple(action)


        sa = (state, action)

        if sa in self.Qsa:
            self.Qsa[sa] = (self.Qsa[sa] * self.Nsa[sa] + reward) / (self.Nsa[sa] + 1)
            self.Nsa[sa] += 1
        else:
            self.Qsa[sa] = reward
            self.Nsa[sa] = 1

    def update_metrics_decision_node(self, state):
        """Update the visit counts for states.
        """
        if state in self.Ns:
            self.Ns[state] += 1
        else:
            self.Ns[state] = 1


    def type_checker(self, observation, reward):
        """Converts the observation and reward from dict and base.Reward type to the correct type if they are not already.

        Args:
            observation (Union[dict, np.ndarray]): Observation to convert.
            reward (Union[float, base.Reward]): Reward to convert.

        Returns:
            (int,np.ndarray): Converted observation.
            (float): Converted reward.
        """
        if isinstance(observation, dict) and 'state' in observation:
            observation = observation['state']

        if isinstance(observation, np.ndarray):
            observation = tuple(observation)
        if isinstance(reward, base.Reward):
            reward = reward.reward

        #DEBUGGING ASSERTION
        assert not isinstance(observation, dict), "Observation is still a dict after type checking."
        return observation, reward
    

    def best_child(self,v):
        """Find the best child nodes based on the UCT value.

        This method is only called for decision nodes.

        Args:
            exploration_constant (_type_, optional): _description_. Defaults to math.sqrt(2).

        Returns:
            Node: The best child node based on the UCT value.
            action: The action that leads to the best child node.
        """
        
        best_value = -np.inf
        best_nodes = []
        children = v.children
        for child in children:
            sa = (child.parent.state, child.action)
            if sa in self.Qsa:
                ucb_value = self.Qsa[sa] + self.c * np.sqrt(
                    np.log(self.Ns.get(sa[0], 1)) / self.Nsa[sa])
            else:
                ucb_value = self.c * np.sqrt(
                    np.log(self.Ns.get(sa[0], 1)) / 1)  # Assume at least one visit
                ucb_value = np.inf

            if ucb_value > best_value:
                best_value = ucb_value
                best_nodes = [child]
            elif ucb_value == best_value:
                best_nodes.append(child)

        return random.choice(best_nodes) if best_nodes else None
    
    def best_action(self,v):
        """Select the best action based on the Q values of the state-action pairs.
        Returns:
            best_action(int): best action to)
        """
        best_action = None
        best_avg_value = -np.inf

        s = v.state # root is Type[Node] 

        # Iterate through all possible actions from this state
        for a in range(self.env.action_space.n): 
            sa = (s, a)  # Create a state-action pair
            # Check if this state-action pair has been explored
            if sa in self.Qsa and sa in self.Nsa and self.Nsa[sa] > 0:
                avg_value = self.Qsa[sa] / self.Nsa[sa]  # Calculate average value
                if avg_value > best_avg_value:
                    best_avg_value = avg_value
                    best_action = a

        # Ensure a valid action is selected, even if no action has been explored
        if best_action is None and self.possible_actions:
            best_action = np.random.choice(self.possible_actions)

        return best_action
    
    def act(self, observation, env):
        """
        Decide on an action using the MCTS search, reinitializing the tree structure.

        Args:
            observation (Union[int, np.ndarray]): The current state or observation of the environment.

        Returns:
            int: The selected action.
        """

        observation,_ = nsg.utils.type_mismatch_checker(observation=observation,reward=None)
        if isinstance(observation, np.ndarray):
            observation = tuple(observation)

        # Reinitialize the instance by calling __init__
        self.__init__(env, observation, self.d, self.m, self.c, self.gamma)

        # Perform MCTS search to determine the best action
        best_action, _ = self.search()

        return best_action


if __name__ == "__main__":


    env = gym.make("CartPole-v1",max_episode_steps=500)
    scheduler  = nsg.schedulers.ContinuousScheduler()
    update_fn= nsg.update_functions.NoUpdate(scheduler=scheduler)
    env = nsg.wrappers.NSClassicControlWrapper(env,tunable_params={"masspole":update_fn})

    ########### EXAMPLE USAGE ################
    # env = gym.make("FrozenLake-v1",max_episode_steps=100)
    # scheduler  = nsg.schedulers.ContinuousScheduler()
    # update_fn= nsg.update_functions.DistributionNoUpdate(scheduler=scheduler)
    # env = nsg.wrappers.NSFrozenLakeWrapper(env,tunable_params={"P":update_fn})
    decision_times  = []


    for i in range(1):    
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0 
        max_steps = 500
        mcts_agent = MCTS(env,obs,15,50,1.44,0.999)
        reward_list = []
        while not done and not truncated and step < max_steps:
            a = mcts_agent.act(obs,env)
            obs,reward,done,truncated,info = env.step(a)
            reward_list.append(reward.reward)
            
            # decision_times.append(time.time()-start)

            step+=1
            if step%10 == 0:
                print("Step ",step)
    print("Reward ",np.sum(reward_list))




