import gymnasium as gym
import numpy as np
from copy import deepcopy
import ns_gym as nsg
from ns_gym import base
import random
from collections import defaultdict
import torch


"""
MCTS with Chance Nodes to handle stochastic environments. This implementation used a global table to store the Q values and visit counts for state-action pairs and states. Compatible with OpenAI Gym environments.
"""

class DecisionNode:
    """
    Decision node class, labelled by a state.
    """
    def __init__(self, parent, state, weight, is_terminal, reward,value):
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

        if isinstance(state, np.ndarray):
            state = tuple(state)
        self.state = state
        self.weight = weight  # Ground truth probability to occur given the parent (state-action pair) -- given by the environment
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.children_priors = []
        self.value = value # value of state predicted by the neural network -- equal to reward if terminal
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


def stable_normalizer(x,temp):
    x  = (x/(np.max(x)))**temp
    return np.abs(x/(np.sum(x)))

class MCTS:
    """MCTS with Chance Nodes for alphazero . Compatible with OpenAI Gym environments.
        Selection and expansion are combined into the "treepolicy method"
        The rollout/simluation is the "default" policy. 
    """
    def __init__(self,env:gym.Env,state,model,d,m,c,gamma) -> None:
        """
        Args:
            env (gym.Env): The environment to run the MCTS on.
            state (Union[int, np.ndarray]): The state to start the MCTS from.
            model (nn.Module): The neural network model to use for the AlphaZero agent.
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
        self.env = env # This is the current state of the mdp
        self.d = d # depth #TODO icorportae this into simulation depth
        self.m = m # number of simulations
        self.c = c # exploration constant
        if isinstance(state,base.Observation):
            state = state.state



        if isinstance(env.action_space,gym.spaces.Discrete):
            self.possible_actions = [x for x in range(env.action_space.n)]
        else:
            raise ValueError("Only Discrete action spaces are supported")
        
        self.gamma = gamma        
        
        self.Qsa = {}  # stores Q values for s,a pairs, defaults to Qsa of 0
        self.Nsa = {}  # stores visit counts for s,a pairs, default to Nsa of 0
        self.Ns = {} # stores visit counts for states, default to Ns of 0
        self.P = {} # stores the prior probabilities of the actions for a given state
        self.W = {} # unnormalized cumulative rewards for each state


        ### Deep Learning Model
        self.model = model
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # self.model = self.model.to(self.device)

        ### ROOT NODE ####
        self.v0 = DecisionNode(parent=None,state=state,weight=1,is_terminal=False,reward=0,value=0)
        action_probs,val = self._evaluate_decision_node(self.v0)
        self.v0.value = val
        self.v0.children_priors = action_probs
        self.v0.children = [ChanceNode(parent=self.v0,action=a) for a in self.possible_actions]

        for a in self.possible_actions:
            sa = (self.v0.state,a)
            self.Qsa[sa] = self.v0.value
            self.W[sa] = 0
            self.Nsa[sa] = 0

        self.Ns[self.v0.state] = 0



    def search(self):
        """Do the MCTS by doing m simulations from the current state s. 
        After doing m simulations we simply choose the action that maximizes the estimate of Q(s,a)

        Returns:
            best_action(int): best action to take
            action_values(list): list of Q values for each action.
        """
        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deep copy of of the og env at the root nod 
            expanded_node = self._tree_policy(self.v0) # get the leaf node from the tree policy
            self._backpropagation(expanded_node.value,expanded_node) # update the Q values and visit counts for the nodes in the path from the root to the leaf 


    def return_results(self,temp):
        """Return the results of the MCTS search after it is finished.
        Args:
            temp (float): Parameter for getting the pi values.
        Returns:
            state (np.ndarray): Game state, usually as a 1D np.ndarray.
            pi (np.ndarray): Action probabilities from this state to the next state as collected by MCTS.
            V_target (float): Actual value of the next state.
        """

        counts = np.array([self.Nsa.get((self.v0.state,a),0) for a in self.possible_actions])
        Q = np.array([self.Qsa.get((self.v0.state,a)) for a in self.possible_actions])
        pi_target = stable_normalizer(counts,temp)
        V_target = np.sum((counts/np.sum(counts))*Q) #calcualte expected value of next state
        best_action = np.argmax(Q)

        if isinstance(self.v0.state, int):
            arr = torch.zeros(self.env.observation_space.n)
            arr[self.v0.state] = 1
            arr = arr.unsqueeze(0)
            state = arr
        else:
            state = self.v0.state
 
        return state, pi_target, V_target, best_action

    def _tree_policy(self, node) -> ChanceNode:
        """Tree policy for MCTS. Traverse the tree from the root node to a leaf node.
        Args:
            node (DecisionNode): The root node of the tree.
        Returns:
            ChanceNode: The leaf node reached by the tree policy.
        """

        while True:
            
            # select the best child node based on the UCT value

            action = self._best_action(node) # select step based on UCT value
            chance_node = node.children[action] # select the child node based on the action
            node,backprop = self._expand_chance_node(chance_node) # step through the environment to get the next state if the node exists keep going else return the node. 

            if node.is_terminal or backprop:
                # return the decision node if is terminal or if it has no children. 
                assert(type(node)==DecisionNode)
                return node
            
    
    def _evaluate_decision_node(self,v:DecisionNode):
        """Evaluate the value of a decision node using the neural network model. (Replaces random rollout with a neural network.)

        Args:
            v (DecisionNode): Leaf node to evaluate.

        Returns:
            action_probs (np.ndarray): Action probabilities for all legal moves in from current state.
            val (float): Value of the current state.
        """

        if v.is_terminal:
            return None,v.reward
        
        with torch.no_grad():
            s = self._network_input_checker(v.state)
            action_probs, val = self.model(s)

        action_probs = action_probs.cpu().numpy().flatten()
        val = val.cpu().numpy().flatten()
        val = val[0]

        return action_probs,val

    def _expand_decision_node(self,v:DecisionNode,action):
        """Expand the tree by adding a new decision node to the tree.
        """
        assert(type(v) == DecisionNode)
        if v.is_terminal:
            return v
        
        # for a in self.possible_actions:
        #     new_node = ChanceNode(parent=v,action=a)
        #     v.children.append(new_node)

        new_node = ChanceNode(parent=v,action=action)
        v.children.append(new_node)

        return new_node
    
    def _expand_chance_node(self,v:ChanceNode):
        """Expand the tree by adding a new chance node to the tree.

        Args:
            v (ChanceNode): The node to expand.
        Returns:
            new_node (DecisionNode): The new node added to the tree.

        """
        assert(type(v) == ChanceNode)
        action = v.action
        obs,reward,term,_,info = self.sim_env.step(action)
        obs,reward = self.type_checker(obs,reward)

        existing_child = [child for child in v.children if child.state == obs]
        if existing_child:
            return existing_child[0],False
        else:
            if "prob" in info:
                w = info["prob"]
            else:
                w = 1    

            new_node = DecisionNode(parent=v,state=obs,weight=w,is_terminal=term,reward=reward,value=0)
            action_probs, value = self._evaluate_decision_node(new_node)
            if term: #If we expand to a terminal state, we set the value to the reward of the terminal state.
                value = reward
            new_node.value = value
            new_node.children_priors = action_probs

            for a in self.possible_actions:
                sa = (new_node.state,a)
                self.Qsa[sa] = self.Qsa.get(sa,new_node.value)   
                self.Nsa[sa] = self.Nsa.get(sa,0)
                self.W[sa] = self.W.get(sa,0)
                new_node.children.append(ChanceNode(parent=new_node,action=a))
            
            v.children.append(new_node)
            return new_node,True

    def _backpropagation(self,R,v,depth=0):
        """Backtrack to update the number of times a node has beenm visited and the value of a node untill we reach the root node. 
        """

        assert(type(v) == DecisionNode)
        assert(v.value == R) 

        while v.parent:
            R = v.value + self.gamma * R
            chance_node = v.parent
            self._update_metrics_chance_node(v.state,chance_node.action,R)
            v = chance_node.parent
            self._update_metrics_decision_node(v.state)

    def _update_metrics_chance_node(self, state, action, reward):
        """Update the Q values and visit counts for state-action pairs and states.

        Args:
            state (Union[int,]): _description
            action (_type_): _description_
            reward (_type_): _description_
        """
        if isinstance(state, np.ndarray):
            state = tuple(state)
        sa = (state, action)
        if sa in self.Qsa:
            self.Nsa[sa] = self.Nsa.get(sa,0) + 1
            self.W[sa] = self.W.get(sa,0) + reward
            # self.Qsa[sa] = (self.Qsa[sa] * self.Nsa.get(sa) + reward) / (self.Nsa.get(sa,0) + 1)
            self.Qsa[sa] = self.W[sa] / self.Nsa.get(sa)
   
        else:
            self.W[sa] = reward
            self.Nsa[sa] = 1
            self.Qsa[sa] = self.W[sa] / self.Nsa.get(sa)

    def _update_metrics_decision_node(self, state):
        """Update the visit counts for states.
        """
        if state in self.Ns:
            self.Ns[state] += 1
        else:
            self.Ns[state] = 1


    def type_checker(self, observation, reward):
        """Converts the observation and reward from base.Observation and base.Rewars type to the correct type if they are not already.

        Args:
            observation (_type_): Observation to convert.
            reward (_type_): Reward to convert.

        Returns:
            (int,np.ndarray): Converted observation.
            (float): Converted reward.
        """
        if isinstance(observation, base.Observation):
            observation = observation.state
        if isinstance(observation, np.ndarray):
            observation = tuple(observation)
        if isinstance(reward, base.Reward):
            reward = reward.reward
        return observation, reward
    

    # def best_child(self,v):
    #     """Find the best child nodes based on the UCT value.

    #     This method is only called for decision nodes.

    #     Args:
    #         v (DecisionNode): The parent node.

    #     Returns:
    #         Node: The best child node based on the UCT value.
    #         action: The action that leads to the best child node.
    #     """
    #     assert(type(v) == DecisionNode)
        
    #     best_value = -np.inf
    #     best_nodes = []
    #     children = v.children # list of ChanceNodes
    #     priors = v.children_priors
    #     assert(len(children) == len(priors))

    #     for child,prior in zip(children,priors):
    #         sa = (child.parent.state, child.action)
    #         if self.Nsa.get(sa,0) == 0:
    #             ucb_value = np.inf
    #         else:
    #             ucb_value = self.Qsa.get(sa,child.parent.value) + self.c * prior* np.sqrt(np.log(self.Ns.get(sa[0], 0)) / (1+self.Nsa.get(sa,0))) # Find the UCB value to include the priors  FIXME

    #         if ucb_value > best_value:
    #             best_value = ucb_value
    #             best_nodes = [child]
    #         elif ucb_value == best_value:
    #             best_nodes.append(child)

    #     return random.choice(best_nodes) if best_nodes else None
    
    def _best_action(self,v):
        """Select the best action based on the Q values of the state-action pairs.
        Returns:
            best_action(int): best action to)
        """
        assert(type(v) == DecisionNode)
        best_action = None
        best_value = -np.inf

        state = v.state

        for action,prior in zip(self.possible_actions,v.children_priors):
            sa = (state, action)
            ucb_value = self.Qsa[sa] + self.c * prior * np.sqrt(1 + self.Ns.get(state,0)) / (1+self.Nsa.get(sa,0))
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action

        return best_action

    
    def _network_input_checker(self, x):
        """Make sure the input to the neural network is in the correct format
        """

        if isinstance(x, base.Observation):
            x = x.state

        s = x 
        if type(s) == int:
            arr = torch.zeros(self.env.observation_space.n)
            arr[s] = 1
            arr = arr.unsqueeze(0)
            s = arr

        if not isinstance(s, torch.Tensor):
            s = torch.Tensor(x)

        s = s.to(self.device)
        return s
    
    def _progressive_widening(self):
        raise NotImplementedError("Progressive widening not implemented yet")
    

    




if __name__ == "__main__":
    pass
