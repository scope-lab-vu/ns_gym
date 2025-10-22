import ns_gym as nsg
import numpy as np
from ns_gym import base

class PAMCTS(base.Agent):
    """ Policy augmented MCTS algorithm
    Uses a convex combination of DDQN policy values and MCTS values to select actions.

    Args:
        alpha (float): PAMCTS convex combination parameter
        env (gym.Env): Gymnasium style environment object
        mcts_iter (int): Total number of MCTS iterations
        mcts_search_depth (int): MCTS search depth
        mcts_discount_factor (float): MCTS discount factor
        mcts_exploration_constant (float): UCT exploration constant `c`
        state_space_size (int): Size of environment state space. For Q-value networks.
        action_space_size (int): Size of environment action space. For Q-value networks.
        DDQN_model (torch.NN, optional): DDQN torch neural network object . Defaults to None.
        DDQN_model_path (str, optional): Path to DDQN model weights. Defaults to None.
    """
    def __init__(self,
                 alpha,
                 mcts_iter,
                 mcts_search_depth,
                 mcts_discount_factor,
                 mcts_exploration_constant,
                 state_space_size,
                 action_space_size,
                 DDQN_model = None,
                 DDQN_model_path = None,
                 seed = 0
                 ) -> None:
     
        self.alpha = alpha
        self.mcts_iter = mcts_iter
        self.mcts_search_depth = mcts_search_depth
        self.mcts_discount_factor = mcts_discount_factor
        self.mcts_exploration_constant = mcts_exploration_constant
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.DDQN_model = DDQN_model
        self.DDQN_model_path = DDQN_model_path

        if self.DDQN_model is None or self.DDQN_model_path is None:
            raise ValueError("DDQN model object and  path to model weights must be provided")
        
        self.DDQN_agent = nsg.benchmark_algorithms.DDQN.DQNAgent(self.state_space_size,self.action_space_size,seed = seed,model=self.DDQN_model,model_path=self.DDQN_model_path)        
    

    def search(self,state,env,normalize=True):
        self.mcts_agent = nsg.benchmark_algorithms.MCTS(env,state,d=self.mcts_search_depth,m=self.mcts_iter,c=self.mcts_exploration_constant,gamma=self.mcts_discount_factor)
        mcts_action,mcts_action_values = self.mcts_agent.search()
        ddqn_action,ddqn_action_values = self.DDQN_agent.search(state)

        ddqn_action_values = np.array(ddqn_action_values).astype(np.float32)
        mcts_action_values = np.array(mcts_action_values).astype(np.float32) 
        if normalize:
            epsilon = 1e-8
            ddqn_action_values = (ddqn_action_values - np.min(ddqn_action_values))/(np.max(ddqn_action_values) - np.min(ddqn_action_values) + epsilon)
            mcts_action_values = (mcts_action_values - np.min(mcts_action_values))/(np.max(mcts_action_values) - np.min(mcts_action_values) + epsilon)

        hybrid_action_values = self._get_pa_uct_score(self.alpha,ddqn_action_values,mcts_action_values)
        return np.argmax(hybrid_action_values), hybrid_action_values
    
    def act(self,state,env,normalize=True):
        action,_  = self.search(state,env,normalize)
        return action
    
    def _get_pa_uct_score(self,alpha, policy_value, mcts_return):

        hybrid_node_value = (alpha * policy_value) + ((1.0 - alpha) * mcts_return)
        return hybrid_node_value  # + self.C * np.sqrt(np.log(self.parent.visits)/self.visits)
    
    

if __name__ == "__main__":
    pass
