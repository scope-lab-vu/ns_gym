import math
import random
#from nsbridge_simulator.nsbridge_v0 import NSBridgeV0 as model
from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
import pickle
from BayesianNeuralNetwork import *
import autograd.numpy as np
import utils.distribution as distribution
from collections import Counter
import time
from ns_gym import wrappers, schedulers, update_functions
import gymnasium as gym
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
        self.bnn_num_samples = 1000  # number of samples of network weights drawn to get each BNN prediction
        self.bnn_batch_size = 32
        self.bnn_v_prior = 3  # Prior variance on the BNN parameters
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
    bnn_cache = {}
    basemodel_aleatoric = []
    newmodel_aleatoric = []
    basemodel_epistemic = []
    newmodel_epistemic = []
    def __init__(self, state, time, task, danger, action=None, parent=None, node_type="decision"):
        self.state = state
        self.action = action  # Action taken or outcome for chance node
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.type = node_type  # "decision" or "chance"
        self.probabilities = [] if self.type == "chance" else None
        self.possible_actions = [0, 1, 2, 3]
        self.discount_factor = 0.9999
        self.time = time
        self.np_random = np.random.RandomState()
        self.task = task
        self.danger = danger

    def get_transition(self,bnn_samples, BNN1, state, action):
        state_count = []
        samples = bnn_samples[:, 0, :]
        for j in samples.reshape(-1, 2):
            state_count.append(BNN1.task._NSFrozenLakeV0__decode_state(j, state, action))
        state_counts = Counter(state_count)  # Assuming state_count is defined somewhere in your code

        # Calculate total transitions
        total_transitions = sum(state_counts.values())

        # Initialize a list of zeros for all states
        bnn_transitions1 = [0.0] * 16

        # Fill in the transition probabilities for the states in state_counts
        for s, c in state_counts.items():
            bnn_transitions1[s] = c / total_transitions
        return bnn_transitions1

    def get_bnn_prediction(self, state, action, BNN1, weight_set1, BNN2, weight_set2, isi):
        # Check cache before BNN prediction
        cache_key = (state, action)
        if cache_key in Node.bnn_cache:
            return Node.bnn_cache[cache_key]
        # If not in cache, query the BNNs
        state1 = BNN1.task._NSFrozenLakeV0__encode_state(state)
        confidence_check = True
        EU1 = []
        EU2 = []
        aug_state1 = np.hstack([state1, self.__encode_action(action), weight_set1]).reshape((1, -1))
        aug_state2 = np.hstack([state1, self.__encode_action(action), weight_set2]).reshape((1, -1))
        _, samples1, epistemic_uncertainties1, aleatoric_uncertainties1 = BNN1.network.feed_forward_distribution(
            aug_state1)
        EU1.append(np.sum(epistemic_uncertainties1))
        _, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = BNN2.network.feed_forward_distribution(
            aug_state2)
        EU2.append(np.sum(epistemic_uncertainties2))
        transition1 = self.get_transition(samples1, BNN1, state, action)
        transition2 = self.get_transition(samples2, BNN1, state, action)
        Node.basemodel_epistemic.append(np.mean(EU1))
        Node.newmodel_epistemic.append(np.mean(EU2))
        pessimistic1 = self.pessimistic_sample(transition1, state, action, isi)
        # Store the results in the cache

        Node.bnn_cache[cache_key] = (
            transition1, samples1, pessimistic1, np.mean(EU1), np.mean(Node.basemodel_aleatoric), transition2, samples2, pessimistic1,
            np.mean(EU2), np.mean(Node.newmodel_aleatoric), confidence_check)
        return transition1, samples1, pessimistic1, np.mean(EU1), np.mean(Node.basemodel_aleatoric), transition2, samples2, pessimistic1, np.mean(EU2), np.mean(Node.newmodel_aleatoric), confidence_check
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

    def pessimistic_sample(self, w0, current_state, curren_action, isi):
        v0 = np.zeros(len(w0))
        for i, j in enumerate(w0):
            if j > 0:
                v0[i] = self.task.instant_reward_byindex(i)
        if np.all(v0 >= 0):
            return w0
        else:
            count = 0
            w_worst = np.zeros(len(w0))
            w_worst[np.argmin(v0)] = 1.0
            if not self.danger:
                return w_worst
            c = 1
            # Using numpy to find indices where rs is 1
            rs = np.array(self.task.reachable_states(current_state, curren_action))
            reachable_states = np.where(rs == 1)[0]
            d = self.task.distances_matrix(reachable_states.tolist())


            # Convert w0 and w_worst to numpy arrays before slicing
            w0_np = np.array(w0)
            w_worst_np = np.array(w_worst)
            w0_dis = w0_np[reachable_states]
            w_worst_dis = w_worst_np[reachable_states]

            wass_value = distribution.wass_dual(w0_dis, w_worst_dis, d)
            if wass_value <= c:
                return w_worst
            lbd = c / wass_value
            w = (1.0 - lbd) * w0_dis + lbd * w_worst_dis
            # Using numpy advanced indexing to assign values
            new_w = np.zeros(len(w0))
            new_w[reachable_states] = w
        return new_w.tolist()


    def is_terminal(self, bnn1, state):
        reward = self.task.instant_reward_byindex(state)
        return reward == 1 or reward == -1

    def expand(self, BNN1, weight_set1, BNN2, weight_set2, threshold, training_started, isi):
        if self.is_terminal(BNN1, self.state):
            return self
        if self.is_decision_node():
            for action in self.get_possible_actions():
                # Create chance nodes for each possible action
                child_node = Node(self.state, self.time, self.task, self.danger, action=action, parent=self, node_type="chance")
                self.children.append(child_node)
        else:  # For a chance node
            transition1, samples1, pessimistic1, epistemic_uncertainties1, aleatoric_uncertainties1, transition2, samples2, pessimistic2, epistemic_uncertainties2, aleatoric_uncertainties2, confidence_check = self.get_bnn_prediction(self.state, self.action, BNN1, weight_set1, BNN2, weight_set2, isi)
            if (np.sum(epistemic_uncertainties1) + threshold < np.sum(epistemic_uncertainties2)) or (not training_started):
                child_state = self.categorical_sample(pessimistic1, self.np_random)
            else:
                if aleatoric_uncertainties1 < aleatoric_uncertainties2:
                    gamma = 10000
                    aleatoric_difference = aleatoric_uncertainties2 - aleatoric_uncertainties1
                    likelihood = math.exp(-gamma * aleatoric_difference)
                    if aleatoric_difference < 1:
                        sampled_value = np.random.uniform(0, 1)
                        if sampled_value < likelihood:
                            child_state = self.categorical_sample(transition2, self.np_random)
                        else:
                            child_state = self.categorical_sample(pessimistic2, self.np_random)
                    else:
                        child_state = self.categorical_sample(pessimistic2, self.np_random)
                else:
                    child_state = self.categorical_sample(transition2, self.np_random)

            existing_child = self.get_child_with_state(child_state)
            if existing_child:
                child_node = existing_child
            else:
                child_node = Node(child_state, self.time, self.task, self.danger, parent=self, node_type="decision")
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

    def rollout(self, BNN1, weight_set1, BNN2, weight_set2, threshold, training_started, isi):
        current_state = self.state
        cumulative_reward = 0.0
        depth = 0
        done = False
        visited_pairs = set()
        while not done:
            #print("current state:", current_state)
            if self.is_terminal(BNN1, current_state):
                cumulative_reward += self.task.instant_reward_byindex(current_state)
                break
            action = random.choice(self.possible_actions)

            transition1, samples1, pessimistic1, epistemic_uncertainties1, aleatoric_uncertainties1, transition2, samples2, pessimistic2, epistemic_uncertainties2,aleatoric_uncertainties2, confidence_check = self.get_bnn_prediction(
                current_state, action, BNN1, weight_set1, BNN2, weight_set2, isi)

            if (np.sum(epistemic_uncertainties1) + threshold < np.sum(epistemic_uncertainties2)) or (not training_started):
                child_state = self.categorical_sample(pessimistic1, self.np_random)
            else:
                if aleatoric_uncertainties1 < aleatoric_uncertainties2:
                    gamma = 10000
                    aleatoric_difference = aleatoric_uncertainties2 - aleatoric_uncertainties1
                    likelihood = math.exp(-gamma * aleatoric_difference)
                    if aleatoric_difference < 1:
                        sampled_value = np.random.uniform(0, 1)
                        if sampled_value < likelihood:
                            child_state = self.categorical_sample(transition2, self.np_random)
                        else:
                            child_state = self.categorical_sample(pessimistic2, self.np_random)
                    else:
                        child_state = self.categorical_sample(pessimistic2, self.np_random)
                else:
                    child_state = self.categorical_sample(transition2, self.np_random)
            reward = self.task.instant_reward_byindex(child_state)
            cumulative_reward += pow(self.discount_factor, depth) * reward
            if reward == 1 or reward == -1:
                done = True
            depth += 1
            current_state = child_state
        return cumulative_reward

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            if self.is_decision_node:
                result = result * self.discount_factor
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
    def __init__(self, initial_state_coordinate, initial_state_index, bnn1, bnn2, ws1, ws2, time, task, threshold, training_started, danger, visited_pair):
        self.root = Node(initial_state_index, time, task, danger)
        self.task = task
        self.BNN1 = bnn1
        self.BNN2 = bnn2
        self.weight_set1 = ws1
        self.weight_set2 = ws2
        self.time = time
        self.threshold = threshold
        self.training_started = training_started
        self.isi = initial_state_index
        Node.bnn_cache = {}
        #if len(Node.basemodel_aleatoric) > 100:
        #    Node.basemodel_aleatoric = Node.basemodel_aleatoric[-100:]
        if len(Node.basemodel_epistemic) > 100:
            Node.basemodel_epistemic = Node.basemodel_epistemic[-100:]
        Node.newmodel_epistemic = []

    def is_terminal(self, state):
        reward = self.task.instant_reward_byindex(state)
        return reward == 1 or reward == -1

    def search(self, iterations):
        for _ in range(iterations):
            leaf = self.traverse(self.root)  # Traverse till you reach a leaf
            expanded_node = leaf.expand(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
            if expanded_node.is_chance_node():
                # If it's a chance node, expand again to get a decision node for rollout
                expanded_node = expanded_node.expand(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
            result = expanded_node.rollout(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
            expanded_node.backpropagate(result)

    def traverse(self, node):
        while node.children:
            if node.is_decision_node():
                if self.training_started:
                    node = node.best_child(math.sqrt(2))
                else:
                    node = node.best_child(math.sqrt(2))
            elif self.is_terminal(node.state):
                return node
            else: # chance node
                node = node.expand(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
        return node

    def best_action(self):
        for child in self.root.children:
            print("values", child.action, child.value, child.visits, child.value/child.visits)
        return max(self.root.children, key=lambda c: c.visits).action

    @staticmethod
    def __encode_action(action):
        a = np.array([0] * 4)
        a[action] = 1
        return a


#domain = 'discretegrid'
domain = 'frozenlake'
with open('results/frozen_0.7/{}_network_weights_itr_2'.format(domain), 'rb') as f1:
    network_weights1 = pickle.load(f1)
with open('results/frozen_0.7/{}_latent_weights_itr_2'.format(domain), 'rb') as f3:
    latent_weights1 = pickle.load(f3)

bnn_hidden_layer_size = 25
bnn_num_hidden_layers = 3
preset_hidden_params = [{'latent_code': 1}]
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


from multiprocessing import Pool

def train_model(seed, bnn2,weight_set2, best_net_work_error, best_latent_error, local_converge_count):
    with open('data_buffer/{}_{}_exp_buffer'.format(domain, seed), 'rb') as f:
        exp_buffer = pickle.load(f)
    instance_count = 1
    # print(exp_buffer)
    # Create numpy array
    exp_buffer_np = np.vstack(exp_buffer)
    #print("exp buffer:",exp_buffer)
    # Collect the instances that each transition came from
    inst_indices = exp_buffer_np[:, 4]
    inst_indices = inst_indices.astype(int)
    # Group experiences by instance
    # Create dictionary where keys are instance indexes and values are np.arrays experiences
    exp_dict = {}
    for idx in range(instance_count):
        exp_dict[idx] = exp_buffer_np[inst_indices == idx]
    X = np.array([np.hstack([exp_buffer_np[tt, 0], exp_buffer_np[tt, 1]]) for tt in range(exp_buffer_np.shape[0])])
    y = np.array([exp_buffer_np[tt, 3] for tt in range(exp_buffer_np.shape[0])])
    num_dims = 2
    num_actions = 4
    num_wb = 5
    relu = lambda x: np.maximum(x, 0.)
    state_diffs = False
    bnn_learning_rate = 0.00025
    wb_learning_rate = 0.00025
    if local_converge_count >= 2:
        tuples_list = [(0.0001, 0.0001), (0.0001,0.0007), (0.0002,0.0002)]
        chosen_tuple = random.choice(tuples_list)
        bnn_learning_rate = chosen_tuple[0]
        wb_learning_rate = chosen_tuple[1]
    param_set = {
        'bnn_layer_sizes': [num_dims + num_actions + num_wb] + [bnn_hidden_layer_size] * bnn_num_hidden_layers + [
            num_dims * 2],
        'weight_count': num_wb,
        'num_state_dims': num_dims,
        'bnn_num_samples': 1000,
        'bnn_batch_size': 32,
        'num_strata_samples': 5,
        'bnn_training_epochs': 1,
        'bnn_v_prior': 1,
        'bnn_learning_rate': bnn_learning_rate,
        'bnn_alpha': 0.5,
        'wb_num_epochs': 1,
        'wb_learning_rate': wb_learning_rate
    }
    # Initialize latent weights for each instance


    full_task_weights = weight_set2
    # Initialize BNN
    network_training = BayesianNeuralNetwork(param_set, nonlinearity=relu)
    network_training.weights = bnn2.network.weights
    output_network_weights = bnn2.network.weights
    output_latent_weights = weight_set2
    def get_random_sample(start,stop,size):
        indices_set = set()
        while (len(indices_set) < size):
            indices_set.add(np.random.randint(start,stop))
            if len(indices_set) >= stop:
                break
        return np.array(list(indices_set))
    # size of sample to compute error on
    sample_size = 1000
    for i in range(3):
        # Update BNN network weights
        network_training.fit_network(exp_buffer_np, full_task_weights, 0, state_diffs=state_diffs,
                            use_all_exp=True)
        #print('finished BNN update '+str(i))
        #print("checkpoint")
        if i % 1 == 0:
            #get random sample of indices
            sample_indices = get_random_sample(0,X.shape[0],sample_size)
            l2_errors = network_training.get_td_error(np.hstack((X[sample_indices],full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
            if (np.mean(l2_errors) + np.std(l2_errors)) < best_net_work_error:
                best_net_work_error = (np.mean(l2_errors) + np.std(l2_errors))
            best_latent_error = np.mean(l2_errors)
            output_network_weights = network_training.weights
            output_latent_weights = full_task_weights.reshape(full_task_weights.shape[1], )
            mse = measure_T_diff_without_plot(bnn2, network_training,
                                          full_task_weights.reshape(full_task_weights.shape[1], ))
            with open("step_training_record_seed_{}".format(seed), 'a') as f:
                f.write("mean_error_{}_std_error_{}_error_sum_{},T_mse_{},best_error_{},W_{},Wb_{}\n".format(np.mean(l2_errors), np.std(l2_errors), np.mean(l2_errors)+np.std(l2_errors), mse, best_net_work_error, bnn_learning_rate,wb_learning_rate))
        # Update latent weights
        for inst in np.random.permutation(instance_count):
            full_task_weights[inst,:] = network_training.optimize_latent_weighting_stochastic(
                exp_dict[inst],np.atleast_2d(full_task_weights[inst,:]),0,state_diffs=state_diffs,use_all_exp=True)
        # Compute error on sample of transitions
        if i % 1 == 0:
            #get random sample of indices
            sample_indices = get_random_sample(0,X.shape[0],sample_size)
            l2_errors = network_training.get_td_error(np.hstack((X[sample_indices],full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
            if (np.mean(l2_errors) + np.std(l2_errors)) < best_net_work_error:
                best_net_work_error = (np.mean(l2_errors) + np.std(l2_errors))
            best_latent_error = np.mean(l2_errors)
            output_network_weights = network_training.weights
            output_latent_weights = full_task_weights.reshape(full_task_weights.shape[1], )
            mse = measure_T_diff_without_plot(bnn2, network_training,
                                              full_task_weights.reshape(full_task_weights.shape[1], ))
            with open("step_training_record_seed_{}".format(seed), 'a') as f:
                f.write("mean_error_{}_std_error_{}_error_sum_{},T_mse_{},best_error_{},W_{},Wb_{}\n".format(np.mean(l2_errors), np.std(l2_errors), np.mean(l2_errors)+np.std(l2_errors), mse, best_net_work_error, bnn_learning_rate,wb_learning_rate))
            #print ("After Latent update: iter: {}, Mean Error: {}, Std Error: {}".format(i,np.mean(l2_errors),np.std(l2_errors)))
            # We check to see if the latent updates are sufficiently different so as to avoid fitting [erroneously] to the same dynamics
            #print ("L2 Difference in latent weights between instances: {}".format(np.sum((full_task_weights[0]-full_task_weights[1])**2)))
    with open("step_training_record_seed_{}".format(seed), 'a') as f:
        f.write("session finished\n")
    #latent_weights2 = full_task_weights.reshape(full_task_weights.shape[1], )
    return output_network_weights, output_latent_weights, best_net_work_error, best_latent_error

def measure_T_diff_without_plot(BNN2,training_network,weight_set2):
    action = [0,1,2,3]
    state_list = []
    mse_values = []
    for s in range(16):
        row, col = BNN2.task.to_m(s)
        letter = BNN2.task.desc[row, col]
        if letter != b'H':  # s is not a Hole
            state_list.append(s)
    num_pairs = len(state_list) * len(action)
    cols = 8  # for example, 4 columns
    rows = (num_pairs // cols) + (1 if num_pairs % cols != 0 else 0)

    #fig, axes = plt.subplots(rows, cols, figsize=(40, 5 * rows))
    #i = 0
    for a in action:
        for state in state_list:
            state1 = BNN2.task._NSFrozenLakeV0__encode_state(state)
            aug_state2 = np.hstack([state1, __encode_action(a), weight_set2]).reshape((1, -1))
            means, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = training_network.feed_forward_distribution(aug_state2)
            state_count = []
            samples2 = samples2[:, 0, :]
            for j in samples2.reshape(-1, 2):
                state_count.append(BNN2.task._NSFrozenLakeV0__decode_state(j,state,a))
            # Example Usage
            from collections import Counter
            state_counts = Counter(state_count)  # Assuming state_count is defined somewhere in your code

            # Calculate total transitions
            total_transitions = sum(state_counts.values())

            # Initialize a list of zeros for all states
            bnn_transitions = [0.0] * 16

            # Fill in the transition probabilities for the states in state_counts
            for s, c in state_counts.items():
                bnn_transitions[s] = c / total_transitions

            true_transitions = BNN2.task.T[state, a, 0]  # Assuming bnn1 is defined somewhere in your code
            mse = np.mean((np.array(bnn_transitions) - np.array(true_transitions))**2)
            mse_values.append(mse)  # Append the MSE to the list


    average_mse = np.mean(mse_values)
    return average_mse
def measure_T_diff(BNN2,weight_set2, seed, count):
    action = [0,1,2,3]
    state_list = []
    mse_values = []
    for s in range(16):
        row, col = BNN2.task.to_m(s)
        letter = BNN2.task.desc[row, col]
        if letter != b'H':  # s is not a Hole
            state_list.append(s)
    num_pairs = len(state_list) * len(action)
    cols = 8  # for example, 4 columns
    rows = (num_pairs // cols) + (1 if num_pairs % cols != 0 else 0)

    #fig, axes = plt.subplots(rows, cols, figsize=(40, 5 * rows))
    i = 0
    transitions_dict = {}
    for a in action:
        for state in state_list:
            state1 = BNN2.task._NSFrozenLakeV0__encode_state(state)
            aug_state2 = np.hstack([state1, __encode_action(a), weight_set2]).reshape((1, -1))
            means, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = BNN2.network.feed_forward_distribution(aug_state2)
            state_count = []
            samples2 = samples2[:, 0, :]
            for j in samples2.reshape(-1, 2):
                state_count.append(BNN2.task._NSFrozenLakeV0__decode_state(j,state,a))
            # Example Usage
            from collections import Counter
            state_counts = Counter(state_count)  # Assuming state_count is defined somewhere in your code

            # Calculate total transitions
            total_transitions = sum(state_counts.values())

            # Initialize a list of zeros for all states
            bnn_transitions = [0.0] * 16

            # Fill in the transition probabilities for the states in state_counts
            for s, c in state_counts.items():
                bnn_transitions[s] = c / total_transitions

            true_transitions = BNN2.task.T[state, a, 0]  # Assuming bnn1 is defined somewhere in your code
            mse = np.mean((np.array(bnn_transitions) - np.array(true_transitions))**2)
            mse_values.append(mse)  # Append the MSE to the list

            if state not in transitions_dict:
                transitions_dict[state] = {}

            # Update the nested dictionary for the current action
            transitions_dict[state][a] = (bnn_transitions, np.sum(epistemic_uncertainties2), np.sum(aleatoric_uncertainties2))
    average_mse = np.mean(mse_values)
    return average_mse, transitions_dict

def run_task(seed):
    hipmdp2 = HiPMDP(domain, preset_hidden_params,
                     run_type=run_type,
                     bnn_hidden_layer_size=bnn_hidden_layer_size,
                     bnn_num_hidden_layers=bnn_num_hidden_layers,
                     bnn_network_weights=network_weights1)
    hipmdp2._HiPMDP__initialize_BNN()
    np.random.seed(seed)
    full_task_weights2 = np.random.normal(0., 0.2, (1, 5))
    rng = np.random.default_rng()
    full_task_weights2 = rng.normal(0., 0.1, (1, 5))
    weight_set2 = full_task_weights2.reshape(full_task_weights2.shape[1], )
    #best_full_task_weights2 = np.random.normal(0., 0.2, (1, 5))
    render = True
    cumulative_reward = 0
    cumulative_penalty = 0
    discounted_returns = []
    counts = []
    gamma = 0.99
    episode_buffer_bnn = []
    exp_buffer= []
    count = 0  # Initialize the counter
    task = model()
    env_time = 1
    #task.reset(env_time, seed)
    episode_reward = 0
    discount_factor = 1
    #average_mse = measure_T_diff(hipmdp2, weight_set2, seed, count)
    average_mse = measure_T_diff_without_plot(hipmdp2, hipmdp2.network, weight_set2)
    with open("training_record_seed_{}".format(seed), 'a') as f:
        f.write("transition_difference_{},\n".format(average_mse))
    #print("transition function difference before update:",average_mse)
    threshold = 0.02
    training_started = False
    best_net_work_error = 100
    best_latent_error = 100
    best_mean_error = 100
    last_best_error =100
    local_converge_count = 0
    state_list = []
    visited_pair = []
    for s in range(16):
        row, col = hipmdp2.task.to_m(s)
        letter = hipmdp2.task.desc[row, col]
        if letter != b'H':  # s is not a Hole
            state_list.append(s)
    for s in state_list:
        for a in range(4):
            visited_pair.append((s, a))

    if len(visited_pair) != 0:
        for pair in visited_pair:
            state1 = hipmdp1.task._NSFrozenLakeV0__encode_state(pair[0])
            aug_state1 = np.hstack([state1, __encode_action(pair[1]), weight_set1]).reshape((1, -1))
            _, samples1, epistemic_uncertainties1, aleatoric_uncertainties1 = hipmdp1.network.feed_forward_distribution(
                aug_state1)
            Node.basemodel_aleatoric.append(np.sum(aleatoric_uncertainties1))


        for pair in visited_pair:
            state1 = hipmdp2.task._NSFrozenLakeV0__encode_state(pair[0])
            aug_state1 = np.hstack([state1, __encode_action(pair[1]), weight_set2]).reshape((1, -1))
            _, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = hipmdp2.network.feed_forward_distribution(
                aug_state1)
            Node.newmodel_aleatoric.append(np.sum(aleatoric_uncertainties2))
    all_transitions = {}
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    # env = gym.make("CartPole-v1")
    fl_scheduler = schedulers.ContinuousScheduler()
    fl_update_fn = update_functions.DistributionNoUpdate(fl_scheduler)
    # cartpole_update_fn = update_functions.NoUpdate(fl_scheduler)
    param = {"P": fl_update_fn}
    # param = {"gravity":cartpole_update_fn}
    env = wrappers.NSFrozenLakeWrapper(env, param, change_notification=True, delta_change_notification=True,
                                       in_sim_change=False, initial_prob_dist=[1, 0, 0])
    env.reset()
    done = False
    while not done:
        count += 1  # Increment the counter for each step
        global TRACKED_STATES
        danger = False
        TRACKED_STATES = []
        #state_coordinate = task.observe()
        state_coordinate = task._NSFrozenLakeV0__encode_state(env.s)
        #state_index = env.state.index
        state_index = env.s
        danger_check = 0
        for a in [0,1,2,3]:
            rs = np.array(task.reachable_states(state_index, a))
            rs = np.where(rs == 1)[0]
            for reachable_states in rs:
                if task.instant_reward_byindex(reachable_states) == -1:
                    danger_check += 1
        if danger_check >= 4:
            danger = True
        else:
            danger = False
        mcts_instance = MCTS(state_coordinate, state_index, hipmdp1, hipmdp2, weight_set1, weight_set2, env_time - 1, task, threshold, training_started, danger, visited_pair)
        start_time = time.time()
        mcts_instance.search(5000)
        #for key, value in Node.bnn_cache.items():
        #    transition1 = value[0]
        #    transition2 = value[4]
        #    print(f"Cache Key: {key}, Transition 1: {transition1}, Transition 2: {transition2}")
        end_time = time.time()
        print("time consumption for 1 iteration:", end_time - start_time)
        with open("time_record", 'a') as f:
            f.write("{},\n".format(end_time - start_time))
        best_action = mcts_instance.best_action()
        next_state, reward, done, _, info = env.step(best_action)
        #print(next_state, reward, done)
        reward = reward.reward
        next_state = next_state.state
        episode_buffer_bnn.append(
            np.reshape(np.array([state_coordinate, __encode_action(best_action), reward, task._NSFrozenLakeV0__encode_state(next_state), 0]),
                       [1, 5]))
        with open("steps_record", 'a') as f:
            f.write(f"{count} {state_index} {best_action}\n")
        if len(episode_buffer_bnn) >= 50 and len(episode_buffer_bnn) % 5 == 0:
            training_started = True
            exp_list = np.reshape(episode_buffer_bnn, [-1, 5])
            exp_buffer = []
            for trans_idx in range(len(exp_list)):
                exp_buffer.append(exp_list[trans_idx])
            with open('data_buffer/{}_{}_exp_buffer'.format(domain, seed), 'wb') as f:
                pickle.dump(exp_buffer, f)
            network_weights2, weight_set2, best_net_work_error, best_latent_error = train_model(seed,hipmdp1, full_task_weights2, best_net_work_error, best_latent_error, local_converge_count)
            if best_net_work_error == last_best_error:
                local_converge_count += 1
            else:
                last_best_error = best_net_work_error
                local_converge_count = 0
                hipmdp2 = HiPMDP(domain, preset_hidden_params,
                                 run_type=run_type,
                                 bnn_hidden_layer_size=bnn_hidden_layer_size,
                                 bnn_num_hidden_layers=bnn_num_hidden_layers,
                                 bnn_network_weights=network_weights2)
                hipmdp2._HiPMDP__initialize_BNN()
            Node.newmodel_aleatoric = []
            if len(visited_pair) != 0:
                for pair in visited_pair:
                    state1 = hipmdp2.task._NSFrozenLakeV0__encode_state(pair[0])
                    aug_state1 = np.hstack([state1, __encode_action(pair[1]), weight_set2]).reshape((1, -1))
                    _, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = hipmdp2.network.feed_forward_distribution(
                        aug_state1)
                    Node.newmodel_aleatoric.append(np.sum(aleatoric_uncertainties2))
            average_mse = measure_T_diff_without_plot(hipmdp2,hipmdp2.network,weight_set2)
            with open("training_record_seed_{}".format(seed), 'a') as f:
                f.write("basemodel_epistemic_{},basemodel_aleatoric_{},newmodel_epistemic_{},newmodel_aleatoric_{},transition_difference_{},\n".format(np.mean(Node.basemodel_epistemic), np.mean(Node.basemodel_aleatoric),np.mean(Node.newmodel_epistemic), np.mean(Node.newmodel_aleatoric), average_mse))
        return_average_mse, return_transitions_dict = measure_T_diff(hipmdp2, weight_set2, seed, count)
        all_transitions[count] = return_transitions_dict
        print("*****************************")
        print("seed:", seed)
        episode_reward += reward * discount_factor
        discount_factor *= gamma
        if render:
            #env.render()
            print(env.render())
            time.sleep(0.1)
    with open('all_transitions_0.4.pkl', 'wb') as f:
        pickle.dump(all_transitions, f)
    with open("temp_result", 'a') as f:
        f.write("seed_{},reward_{},count_{},\n".format(seed, reward, count))
    if reward == 1:
        cumulative_reward += reward
    if reward == -1:
        cumulative_penalty += reward
    discounted_returns.append(episode_reward)
    counts.append(count)
    print("seed_{}_end,reward_{}.penalty_{}".format(seed,cumulative_reward, cumulative_penalty))
    return discounted_returns, counts, cumulative_reward, cumulative_penalty

if __name__ == '__main__':
    #for i in range(2):
    with Pool() as pool:
        results = pool.map(run_task,[3])
   #results = pool.map(run_task, [2])

    # Write the discounted returns and counts to a text file
    with open('results_fintune_frozen1_0.7in1_threshold.txt', 'a') as f:
        f.write("Discounted Return, Count\n")
        for r in results:
            for dr, c in zip(r[0], r[1]):
                f.write(f"{dr}, {c}\n")
            print("reward:", r[2])
            print("penalty:", r[3])
