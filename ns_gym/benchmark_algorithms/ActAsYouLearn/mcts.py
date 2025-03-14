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
    def __init__(self, state, action=None, parent=None, node_type="decision"):
        self.state = state
        self.action = action  # Action taken or outcome for chance node
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.type = node_type  # "decision" or "chance"
        self.probabilities = [] if self.type == "chance" else None
        self.possible_actions = [0, 1, 2, 3]
        self.discount_factor = 0.95
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

    def is_terminal(self, bnn1, state):
        reward = bnn1.task.instant_reward_byindex(state)
        return reward == 1 or reward == -1

    def expand(self, BNN1, weight_set1):
        if self.is_terminal(BNN1, self.state):
            return self
        if self.is_decision_node():
            for action in self.get_possible_actions():
                # Create chance nodes for each possible action
                child_node = Node(self.state, action=action, parent=self, node_type="chance")
                self.children.append(child_node)
        else:  # For a chance node
            #print(self.state)
            state1 = BNN1.task._NSFrozenLakeV0__encode_state(self.state)
            #state1 = BNN1.task._NSBridgeV0__encode_state(self.state)
            #print(self.state.shape)
            aug_state1 = np.hstack([state1, self.__encode_action(self.action), weight_set1]).reshape((1, -1))
            child_state, _, _, _ = BNN1.network.feed_forward_distribution(aug_state1)
            child_state = child_state.flatten()
            child_state = BNN1.task._NSFrozenLakeV0__decode_state(child_state, self.state, self.action)
            existing_child = self.get_child_with_state(child_state)
            if existing_child:
                child_node = existing_child
            else:
                child_node = Node(child_state, parent=self, node_type="decision")
                self.children.append(child_node)
        return child_node

    """ Simulate until a terminal state """

    def rollout(self, bnn1, ws1):
        #current_state = self.state.flatten()
        current_state = self.state
        cumulative_reward = 0.0
        depth = 0
        done = False
        visited_pairs = set()
        while not done:
            if self.is_terminal(bnn1, self.state):
                #print("end1",self.state)
                cumulative_reward += bnn1.task.instant_reward_byindex(self.state)
                break
            #elif depth >= 20:
            #    break
            unvisited_actions = [action for action in self.possible_actions if
                                 (current_state, action) not in visited_pairs]

            if not unvisited_actions:  # If all actions are already visited
                break

            action = random.choice(unvisited_actions)
            visited_pairs.add((current_state, action))
            #state1 = bnn1.task._NSBridgeV0__encode_state(current_state)
            state1 = bnn1.task._NSFrozenLakeV0__encode_state(current_state)
            #print("rollout",current_state.shape)
            #print(state1)
            aug_state1 = np.hstack([state1, self.__encode_action(action), ws1]).reshape(
                (1, -1))
            child_state, _, _, _ = bnn1.network.feed_forward_distribution(
                aug_state1)
            reward = bnn1.task.instant_reward_bycoordinate(child_state.flatten())
            cumulative_reward += pow(self.discount_factor, depth) * reward
            if reward == 1 or reward == -1:
                #print("end2",bnn1.task._NSFrozenLakeV0__decode_state(child_state.flatten()))
                done = True
            depth += 1
            current_state = bnn1.task._NSFrozenLakeV0__decode_state(child_state.flatten(), current_state, action)
            #current_state = bnn1.task._NSBridgeV0__decode_state(child_state.flatten(), current_state, action)
        return cumulative_reward

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

    def uct_value(self, parent_visits, exploration_constant=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self, exploration_constant=math.sqrt(2)):
        return max(self.children, key=lambda c: c.uct_value(self.visits, exploration_constant))

    @staticmethod
    def __encode_action(action):
        a = np.array([0] * 4)
        a[action] = 1
        return a

class MCTS:
    def __init__(self, initial_state_coordinate, initial_state_index, bnn1, ws1):
        #self.root = Node(bnn1.task._NSBridgeV0__decode_state(initial_state_coordinate, initial_state_index, -1))
        self.root = Node(bnn1.task._NSFrozenLakeV0__decode_state(initial_state_coordinate, initial_state_index, -1))
        self.BNN1 = bnn1
        self.weight_set1 = ws1

    def is_terminal(self, state):
        reward = self.BNN1.task.instant_reward_byindex(state)
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
        return max(self.root.children, key=lambda c: c.visits).action


#domain = 'discretegrid'
domain = 'frozenlake'
#with open('results/laten1_var_full/{}_network_weights'.format(domain), 'rb') as f1:
#    network_weights1 = pickle.load(f1)
#with open('results/laten1_var_full/{}_latent_weights'.format(domain), 'rb') as f3:
#    latent_weights1 = pickle.load(f3)
with open('results/{}_network_weights_itr_4'.format(domain), 'rb') as f1:
    network_weights1 = pickle.load(f1)
with open('results/{}_latent_weights_itr_4'.format(domain), 'rb') as f3:
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

hipmdp1._HiPMDP__initialize_BNN()
weight_set1 = latent_weights1.reshape(latent_weights1.shape[1],)
task = model()
task.reset(1)
#state = task.observe()
#start_state = state
#root = start_state
#mcts_instance = MCTS(root, hipmdp1, weight_set1)
#mcts_instance.search(1000)
#best_action = mcts_instance.best_action()
#print(f"Best action from start: {best_action}")
render = True
cumulative_reward = 0
for i in range(10):
    task = model()
    task.reset(1)
    while not task.is_done():
        print("time",task.time)
        state_coordinate = task.observe()
        state_index = task.state.index
        mcts_instance = MCTS(state_coordinate, state_index, hipmdp1, weight_set1)
        mcts_instance.search(1000)
        best_action = mcts_instance.best_action()
        next_state, reward, done, _ = task.step(best_action)
        if render:
            import time
            print(task.t)
            task.render()
            time.sleep(0.1)
    if reward == 1:
        cumulative_reward += reward
print(cumulative_reward)