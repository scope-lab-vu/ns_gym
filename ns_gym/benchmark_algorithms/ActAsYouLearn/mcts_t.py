import math
import random
#from nsbridge_simulator.nsbridge_v0 import NSBridgeV0 as model
from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
import pickle
from BayesianNeuralNetwork import *
import numpy as np

class HiPMDP(object):
    """
    The HiP-MDP class can be used to:
    - Create a new batch of experience using agent learning a policy modelfree (run_type='modelfree', create_exp_batch=True)
    - Test one of the following methods on a single test instance:
        - Full HiP-MDP with embedded latent weights (run_type='full' and load pretrained bnn_network_weights)
        - Full HiP-MDP with linear top latent weights (run_type='full_linear' and load pretrained bnn_network_weights)
        - Average model (run_type='onesize' and load pretrained bnn_network_weights)
        - Model from scratch (run_type='modelbased')
        - Model-free (run_type='modelfree')
    """

    def __init__(self, domain, preset_hidden_params, run_type='full', ddqn_learning_rate=0.0001,
                 episode_count=500, bnn_hidden_layer_size=25, bnn_num_hidden_layers=2, bnn_network_weights=None,
                 eps_min=0.15, test_inst=None, create_exp_batch=False, num_batch_instances=False, save_results=False,
                 grid_beta=0.23, print_output=False):
        """
        Initialize framework.

        Arguments:
        domain -- the domain the framework will be used on ('grid','hiv', or 'acrobot')
        preset_hidden_params -- List of dictionaries; one dictionary for each instance, where each dictionary contains
            the hidden parameter settings for that instance

        Keyword arguments:
        run_type -- 'full': Constructs a HiP-MDP model through which transfer is facilitated and with which accelerates policy learning,
                    'full_linear': Constructs a HiP-MDP model, but associates the latent weights w_b as a linear weighting of
                        the model features rather than using them as input as is done in the full HiP-MDP model.
                    'modelfree': Learns a policy based solely on observed transitions,
                    'modelbased': builds a model for accelerating policy learning from only the current instance's data
        ddqn_learning_rate -- DQN ADAM learning rate (default=0.0001)
        episode_count -- Number of episodes per instance (default=500)
        bnn_hidden_layer_size -- Number of units in each hidden layer (default=25)
        bnn_num_hidden_layers -- Number hidden layers (default=2)
        bnn_network -- 1-D numpy array of pretrained BNN network weights (default=None)
        eps_min -- Minimum epsilon value for e-greedy policy (default=0.15)
        test_inst -- Index corresponding to the desired test instance; irrelevant when creating an experience batch (default=None)
        create_exp_batch -- Boolean indicating if this framework is for creating an experience batch (default=False)
        num_batch_instances -- number of instances desired in constructing a batch of data to train, default is false to cleanly override if not specified
        grid_beta -- Beta hyperparameter for grid domain governing; a weight on the magnitude of the "drift" (default=0.23)
        print_output -- Print verbose output
        """

        self.__initialize_params()

        # Store arguments
        self.domain = domain
        self.ddqn_learning_rate = ddqn_learning_rate
        self.run_type = run_type
        if self.run_type in ['full', 'full_linear']:
            self.run_type_full = True
        else:
            self.run_type_full = False
        self.preset_hidden_params = preset_hidden_params
        self.bnn_hidden_layer_size = bnn_hidden_layer_size
        self.bnn_num_hidden_layers = bnn_num_hidden_layers
        self.bnn_network_weights = bnn_network_weights
        self.eps_min = eps_min
        self.test_inst = test_inst
        self.create_exp_batch = create_exp_batch
        self.num_batch_instances = num_batch_instances
        self.save_results = save_results
        self.grid_beta = grid_beta
        self.print_output = print_output
        # Set domain specific hyperparameters
        self.__set_domain_hyperparams()
        self.episode_count = episode_count
        # set epsilon step size

    def __initialize_params(self):
        """Initialize standard framework settings."""
        self.instance_count = 1  # number of task instances
        self.episode_count = 500  # number of episodes
        self.weight_count = 5  # number of latent weights
        self.eps_max = 1.0  # initial epsilon value for e-greedy policy
        self.bnn_and_latent_update_interval = 10  # Number of episodes between BNN and latent weight updates
        self.num_strata_samples = 5  # The number of samples we take from each strata of the experience buffer
        self.ddqn_batch_size = 50  # The number of data points pulled from the experience buffer for replay
        self.tau = 0.005  # The transfer rate between our primary DQN and the target DQN
        self.discount_rate = 0.99  # RL discount rate of expected future rewards
        self.beta_zero = 0.5  # Initial bias correction parameter for Importance Sampling when doing prioritized experience replay
        self.bnn_num_samples = 50  # number of samples of network weights drawn to get each BNN prediction
        self.bnn_batch_size = 32
        self.bnn_v_prior = 1.0  # Prior variance on the BNN parameters
        self.bnn_training_epochs = 100  # number of epochs of SGD in each BNN update
        self.num_episodes_avg = 30  # number of episodes used in moving average reward to determine whether to stop DQN training
        self.num_approx_episodes = 500  # number of approximated rollouts using the BNN to train the DQN
        self.state_diffs = False  # Use BNN to predict (s'-s) rather than s'
        self.num_bnn_updates = 3  # number of calls to update_BNN()
        self.wb_learning_rate = 0.0005  # latent weight learning rate
        self.num_batch_updates = 5  # number of minibatch updates to DQN
        self.bnn_alpha = 0.5  # BNN alpha divergence parameter
        self.policy_update_interval = 10  # Main DQN update interval (in timesteps)
        self.target_update_interval = 10  # Target DQN update interval (in timesteps)
        self.ddqn_hidden_layer_sizes = [256, 512]  # DDQN hidden layer sizes
        self.eps_decay = 0.999  # Epsilon decay rate
        self.grad_clip = 2.5  # DDQN Gradient clip by norm
        self.ddqn_batch_size = 50  # DDQN batch size
        # Prioritized experience replay hyperparameters
        self.PER_alpha = 0.2
        self.PER_beta_zero = 0.1
        self.tau = 0.005  # DQN target network update proportion
        self.wb_num_epochs = 100  # number of epochs of SGD in each latent weight update

    def __set_domain_hyperparams(self):
        if self.domain == 'grid':
            self.task = model(beta=self.grid_beta)
        elif self.domain == 'discretegrid':
            self.task = model(time=0)
        else:
            self.task = model()
        # self.var_params = self.task.perturb_params # names of hidden parameters to be varied
        self.num_actions = self.task.num_actions  # number of actions
        if self.domain != 'discretegrid':
            self.num_dims = len(self.task.observe())  # number of state dimensions
        else:
            self.num_dims = len(self.task.observe())
        # print("check!!!!!!!!:", self.task.observe())
        # create set of parameters for each experience replay instantiation


    def __initialize_BNN(self):
        """Initialize the BNN and set pretrained network weights (if supplied)."""
        # Generate BNN layer sizes
        if self.run_type != 'full_linear':
            bnn_layer_sizes = [self.num_dims + self.num_actions + self.weight_count] + [
                self.bnn_hidden_layer_size] * self.bnn_num_hidden_layers + [self.num_dims*2]
        else:
            bnn_layer_sizes = [self.num_dims + self.num_actions] + [
                self.bnn_hidden_layer_size] * self.bnn_num_hidden_layers + [self.num_dims * self.weight_count]
        # activation function
        self.bnn_learning_rate = 0.0005
        relu = lambda x: np.maximum(x, 0.0)
        # Gather parameters
        param_set = {
            'bnn_layer_sizes': bnn_layer_sizes,
            'weight_count': self.weight_count,
            'num_state_dims': self.num_dims,
            'bnn_num_samples': self.bnn_num_samples,
            'bnn_batch_size': self.bnn_batch_size,
            'num_strata_samples': self.num_strata_samples,
            'bnn_training_epochs': self.bnn_training_epochs,
            'bnn_v_prior': self.bnn_v_prior,
            'bnn_learning_rate': self.bnn_learning_rate,
            'bnn_alpha': self.bnn_alpha,
            'wb_learning_rate': self.wb_learning_rate,
            'wb_num_epochs': self.wb_num_epochs
        }
        if self.run_type != 'full_linear':
            self.network = BayesianNeuralNetwork(param_set, nonlinearity=relu)
        else:
            self.network = BayesianNeuralNetwork(param_set, nonlinearity=relu, linear_latent_weights=True)
        # Use previously trained network weights
        if self.bnn_network_weights is not None:
            self.network.weights = self.bnn_network_weights


def __encode_action(action):
    """One-hot encodes the integer action supplied."""
    a = np.array([0] * 4)
    a[action] = 1
    return a


class Node:

    def __init__(self, state, time, task, action=None, parent=None, node_type="decision", mcts=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.type = node_type
        self.possible_actions = [0, 1, 2, 3]
        self.discount_factor = 0.99
        self.time = time
        self.np_random = np.random.RandomState()
        self.task = task
        self.mcts = mcts  # Reference to the MCTS object to access tables
    def is_decision_node(self):
        return self.type == "decision"

    def is_chance_node(self):
        return self.type == "chance"

    def get_possible_actions(self):
        return self.possible_actions

    def get_child_with_state(self, state):
        for child in self.children:
            if np.array_equal(child.state, state):  # or any appropriate equality check for states
                return child
        return None


    def pessimistic_sample(self, w0):
        v0 = np.zeros(len(w0))
        for i, j in enumerate(w0):
            if j > 0:
                v0[i] = task.instant_reward_byindex(i)
        if np.all(v0 >= 0):
            child_state = self.categorical_sample(w0, self.np_random)
        else:
            child_state = np.argmin(v0)
        return child_state


    def is_terminal(self, bnn1, state):
        reward = self.task.instant_reward_byindex(state)
        return reward == 1 or reward == -1

    def expand(self, BNN1, weight_set1):
        if self.is_terminal(BNN1, self.state):
            return self
        if self.is_decision_node():
            for action in self.get_possible_actions():
                # Create chance nodes for each possible action
                child_node = Node(self.state, self.time, self.task, action=action, parent=self, node_type="chance", mcts=self.mcts)
                self.children.append(child_node)
        else:  # For a chance node
            w0 = self.task.T[self.state, self.action, self.time]
            child_state = self.pessimistic_sample(w0)
            if self.state == 4 and self.action == 2:
                TRACKED_STATES.append(child_state)
            # Check if a child with the resulting state already exists
            existing_child = self.get_child_with_state(child_state)
            if existing_child:
                child_node = existing_child
            else:
                child_node = Node(child_state, self.time, self.task,parent=self, node_type="decision", mcts=self.mcts)
                self.children.append(child_node)
        return child_node

    def categorical_sample(self, prob_n, np_random):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np_random.rand()).argmax()

    """ Simulate until a terminal state """

    def rollout(self, bnn1, ws1):
        #current_state = self.state.flatten()
        current_state = self.state
        cumulative_reward = 0.0
        depth = 1
        done = False
        cumulative_reward += self.task.instant_reward_byindex(current_state)
        while not done:
            if self.is_terminal(bnn1, current_state):
                #cumulative_reward += self.task.instant_reward_byindex(current_state)
                break

            action = random.choice(self.possible_actions)
            d = self.task.T[current_state, action, self.time]
            #child_state = self.pessimistic_sample(d)
            child_state = self.categorical_sample(d, self.np_random)
            if current_state == 4 and action == 2:
                TRACKED_STATES.append(child_state)
            reward = self.task.instant_reward_byindex(child_state)
            cumulative_reward += pow(self.discount_factor, depth) * reward
            if reward == 1 or reward == -1:

                done = True
            depth += 1
            current_state = child_state
        #print(cumulative_reward)
        return cumulative_reward

    def backpropagate(self, result):
        # Increment visit count for this node
        self.mcts.increment_visit_count(self)

        # Update Q-value for this node
        current_value = self.mcts.get_q_value(self)
        current_visits = self.mcts.get_visit_count(self)
        #print((1 / current_visits))
        #print( (result - current_value))
        updated_value = current_value + (1 / current_visits) * (result - current_value)
        self.mcts.set_q_value(self, updated_value)

        # Continue backpropagation up the tree
        if self.parent:
            self.parent.backpropagate(result)

    def uct_value(self, exploration_constant=math.sqrt(2)):
        current_visits = self.mcts.get_visit_count(self)
        if current_visits == 0:
            return float('inf')
        parent_visits = self.mcts.get_visit_count(self.parent)
        q_value = self.mcts.get_q_value(self)
        return q_value + exploration_constant * math.sqrt(math.log(parent_visits) / current_visits)

    def best_child(self, exploration_constant=math.sqrt(2)):
        return max(self.children, key=lambda c: c.uct_value(exploration_constant))

    @staticmethod
    def __encode_action(action):
        a = np.array([0] * 4)
        a[action] = 1
        return a


class MCTS:
    def __init__(self, initial_state_coordinate, initial_state_index, bnn1, ws1, time, task):
        self.root = Node(initial_state_index, time, task, mcts=self)
        self.q_values = {}  # Table for Q values
        self.visit_counts = {}  # Table for visit numbers
        self.task = task
        self.BNN1 = bnn1
        self.weight_set1 = ws1
        self.time = time

    # Helper functions to interact with tables
    def get_q_value(self, node):
        return self.q_values.get((node.state, node.action), 0.0)

    def set_q_value(self, node, value):
        self.q_values[(node.state, node.action)] = value

    def get_visit_count(self, node):
        return self.visit_counts.get((node.state, node.action), 0)

    def increment_visit_count(self, node):
        current_visits = self.get_visit_count(node)
        self.visit_counts[(node.state, node.action)] = current_visits + 1

    def is_terminal(self, state):
        reward = self.task.instant_reward_byindex(state)
        return reward == 1 or reward == -1

    def search(self, iterations):
        for _ in range(iterations):
            leaf = self.traverse(self.root)  # Traverse till you reach a leaf
            expanded_node = leaf.expand(self.BNN1, self.weight_set1)
            if expanded_node.is_chance_node():
                # If it's a chance node, expand again to get a decision node for rollout
                expanded_node = expanded_node.expand(self.BNN1, self.weight_set1)
            result = expanded_node.rollout(self.BNN1, self.weight_set1)
            expanded_node.backpropagate(result)

    def traverse(self, node):
        while node.children:
            if node.is_decision_node():
                node = node.best_child()
            elif self.is_terminal(node.state):
                return node
            else: # chance node
                node = node.expand(self.BNN1, self.weight_set1)
        return node

    def best_action(self):
        for child in self.root.children:
            print("values",child.action, self.get_visit_count(child), self.get_q_value(child))
        return max(self.root.children, key=lambda c: self.get_visit_count(c)).action


#domain = 'discretegrid'
domain = 'frozenlake'
#with open('results/laten1_var_full/{}_network_weights'.format(domain), 'rb') as f1:
#    network_weights1 = pickle.load(f1)
#with open('results/laten1_var_full/{}_latent_weights'.format(domain), 'rb') as f3:
#    latent_weights1 = pickle.load(f3)
with open('results/frozen_0.3/{}_network_weights_itr_4'.format(domain), 'rb') as f1:
    network_weights1 = pickle.load(f1)
with open('results/frozen_0.3/{}_latent_weights_itr_4'.format(domain), 'rb') as f3:
    latent_weights1 = pickle.load(f3)
#num_dims = len(task.observe())
#print(task.observe())
#num_actions = task.num_actions
#weight_count = 5
bnn_hidden_layer_size = 25
bnn_num_hidden_layers = 3
#bnn_layer_sizes = [num_dims+num_actions+weight_count] + [bnn_hidden_layer_size]*bnn_num_hidden_layers + [num_dims*2]
preset_hidden_params = [{'latent_code': 2}]
run_type = "full"
hipmdp1 = HiPMDP(domain,preset_hidden_params,
                 run_type=run_type,
                 bnn_hidden_layer_size=bnn_hidden_layer_size,
                 bnn_num_hidden_layers=bnn_num_hidden_layers,
                 bnn_network_weights=network_weights1)


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

hipmdp1._HiPMDP__initialize_BNN()
weight_set1 = latent_weights1.reshape(latent_weights1.shape[1],)
task = model()
task.reset(1)
d = task.T[4, 2, 0]
#print(d)
np_random = np.random.RandomState()
L = []
#for i in range(1000):
#    child_state = categorical_sample(d, np_random)
#    L.append(child_state)
#print(Counter(L))
#state = task.observe()
#start_state = state
#root = start_state
#mcts_instance = MCTS(root, hipmdp1, weight_set1)
#mcts_instance.search(1000)
#best_action = mcts_instance.best_action()
#print(f"Best action from start: {best_action}")
render = True
cumulative_reward = 0
cumulative_penalty = 0
# Initialize the list to store discounted returns
discounted_returns = []

# Discount factor (you can set this to your desired value)
gamma = 0.99

for i in range(10):
    task = model()
    env_time = 1
    seed = i
    task.reset(env_time, seed)
    episode_reward = 0
    discount_factor = 1
    while not task.is_done():
        global TRACKED_STATES
        TRACKED_STATES = []
        state_coordinate = task.observe()
        state_index = task.state.index
        mcts_instance = MCTS(state_coordinate, state_index, hipmdp1, weight_set1, env_time-1, task)
        mcts_instance.search(10000)
        print("*********************************")
        print("iteration", i)
        best_action = mcts_instance.best_action()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        next_state, reward, done, _ = task.step(best_action)
        episode_reward += reward * discount_factor
        discount_factor *= gamma
        if render:
            import time
            task.render()
            time.sleep(0.1)
    if reward == 1:
        cumulative_reward += reward
    if reward == -1:
        cumulative_penalty += reward
    discounted_returns.append(episode_reward)

# Write the discounted returns to a text file
with open('discounted_returns.txt', 'w') as f:
    for r in discounted_returns:
        f.write(str(r) + '\n')
print("reward:",cumulative_reward)
print("penalty:",cumulative_penalty)