import numpy as np
from BayesianNeuralNetwork import *

"""
General Framework for Hidden Parameter Markov Decision Processes (HiP-MDPs) and benchmarks.
"""
#from __future__ import print_function
# import tensorflow as tf
import pickle
from BayesianNeuralNetwork import *
import matplotlib.pyplot as plt

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
        if self.run_type == 'modelfree':
            self.eps_step = (self.eps_max - self.eps_min) / self.episode_count
        else:
            self.eps_step = (self.eps_max - self.eps_min) / self.num_approx_episodes

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
        """Set domain specific hyperparameters."""
        self.standardize_rewards = False
        self.standardize_states = False
        # Acrobot settings
        if self.domain == 'acrobot':
            from acrobot_simulator.acrobot import Acrobot as model
            if self.create_exp_batch:
                if self.num_batch_instances:
                    self.instance_count = self.num_batch_instances
                else:
                    self.instance_count = 8  # number of instances to include in experience batch
            self.max_task_examples = 400  # maximum number of time steps per episode
            self.min_avg_rwd_per_ep = -12  # minimum average reward before stopping DQN training
            self.bnn_learning_rate = 0.00025
            self.num_initial_update_iters = 5  # number of initial updates to the BNN and latent weights
            self.bnn_start = 400  # number of time steps observed before starting BNN training
            self.dqn_start = 400  # number of time steps observed before starting DQN training
        # Grid settings
        elif self.domain == 'grid':
            from grid_simulator.grid import Grid as model
            if self.create_exp_batch:
                if self.num_batch_instances:
                    self.instance_count = self.num_batch_instances
                else:
                    self.instance_count = 2
            self.max_task_examples = 100
            self.min_avg_rwd_per_ep = 980
            self.bnn_learning_rate = 0.00005
            self.num_initial_update_iters = 10
            self.num_approx_episodes = 1000  # Use extra approximated episodes since finding the goal state takes a bit of luck
            self.bnn_start = 100
            self.dqn_start = 1000
            self.wb_num_epochs = 300
            if self.run_type_full:
                self.eps_decay = np.exp(np.log(self.eps_min) / self.num_approx_episodes)
            # self.eps_decay = np.exp(np.log(0.01) / self.num_approx_episodes)
            # self.eps_decay = 0.998
            # In order to learn, model-based from scratch needs some adjustments
            if self.run_type == 'modelbased':
                self.bnn_learning_rate = 0.0005
                self.dqn_start = 400

        # Grid settings
        elif self.domain == 'discretegrid':
            from nsbridge_simulator.nsbridge_v0 import NSBridgeV0 as model
            if self.create_exp_batch:
                if self.num_batch_instances:
                    self.instance_count = self.num_batch_instances
                else:
                    self.instance_count = 2
            self.max_task_examples = 1000
            self.min_avg_rwd_per_ep = 1
            self.bnn_learning_rate = 0.00005
            self.num_initial_update_iters = 10
            self.num_approx_episodes = 1000  # Use extra approximated episodes since finding the goal state takes a bit of luck
            self.bnn_start = 100
            self.dqn_start = 100
            self.wb_num_epochs = 300
            if self.run_type_full:
                # self.eps_decay = np.exp(np.log(self.eps_min)/self.num_approx_episodes)
                self.eps_decay = np.exp(np.log(0.01) / self.num_approx_episodes)
            # In order to learn, model-based from scratch needs some adjustments
            if self.run_type == 'modelbased':
                self.bnn_learning_rate = 0.0005
                self.dqn_start = 400

        elif self.domain == 'frozenlake':
            from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
            if self.create_exp_batch:
                if self.num_batch_instances:
                    self.instance_count = self.num_batch_instances
                else:
                    self.instance_count = 2
            self.max_task_examples = 1000
            self.min_avg_rwd_per_ep = 1
            self.bnn_learning_rate = 0.00005
            self.num_initial_update_iters = 10
            self.num_approx_episodes = 1000  # Use extra approximated episodes since finding the goal state takes a bit of luck
            self.bnn_start = 100
            self.dqn_start = 100
            self.wb_num_epochs = 300
            if self.run_type_full:
                # self.eps_decay = np.exp(np.log(self.eps_min)/self.num_approx_episodes)
                self.eps_decay = np.exp(np.log(0.01) / self.num_approx_episodes)
            # In order to learn, model-based from scratch needs some adjustments
            if self.run_type == 'modelbased':
                self.bnn_learning_rate = 0.0005
                self.dqn_start = 400


        # HIV settings
        elif self.domain == 'hiv':
            from hiv_simulator.hiv import HIVTreatment as model
            if self.create_exp_batch:
                if self.num_batch_instances:
                    self.instance_count = self.num_batch_instances
                else:
                    self.instance_count = 5
            self.max_task_examples = 200
            self.min_avg_rwd_per_ep = 1e15
            self.bnn_learning_rate = 0.00025
            self.num_initial_update_iters = 10
            self.bnn_start = 200
            self.dqn_start = 200
            self.standardize_rewards = True
            self.bnn_alpha = 0.45  # Alpha divergence hyper parameter
            self.bnn_batch_size = 100  # Draw 500 samples total
            self.standardize_states = True
        else:
            raise NameError('invalid domain')
        # Size of buffer for storing batch of experiences
        self.general_bnn_buffer_size = self.instance_count * self.max_task_examples * self.episode_count
        # Size of experience buffer for test instance. Note: all experiences are stored
        self.instance_buffer_size = self.max_task_examples * self.episode_count
        # Size of fictional experience buffer
        self.instance_fictional_buffer_size = self.num_approx_episodes * self.episode_count
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
        self.experience_replay_param_set = {
            'episode_count': self.episode_count,
            'instance_count': self.instance_count,
            'max_task_examples': self.max_task_examples,
            'ddqn_batch_size': self.ddqn_batch_size,
            'num_strata_samples': self.num_strata_samples,
            'PER_alpha': self.PER_alpha,
            'PER_beta_zero': self.PER_beta_zero,
            'bnn_batch_size': self.bnn_batch_size,
            'dqn_start': self.dqn_start,
            'bnn_start': self.bnn_start
        }

    def __get_instance_param_set(self):
        """Get preset hidden parameter setting for this instance."""
        if self.create_exp_batch:
            instance_idx = self.instance_iter
        else:
            instance_idx = self.test_inst
        print("latent", self.preset_hidden_params[instance_idx])
        self.instance_param_set = self.preset_hidden_params[instance_idx]

    def __encode_action(self, action):
        """One-hot encodes the integer action supplied."""
        a = np.array([0] * self.num_actions)
        a[action] = 1
        return a


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


num_dims = 2
num_actions = 4
num_wb = 5
ddqn_learning_rate = 0.0001
episode_count = 10000
bnn_hidden_layer_size = 25
bnn_num_hidden_layers = 3
bnn_network_weights = None
eps_min = 0
test_inst = None
create_exp_batch = True
state_diffs = False
grid_beta = 0.1
relu = lambda x: np.maximum(x, 0.)
param_set = {
    'bnn_layer_sizes': [num_dims+num_actions+num_wb]+[bnn_hidden_layer_size]*bnn_num_hidden_layers+[num_dims],
    'weight_count': num_wb,
    'num_state_dims': num_dims,
    'bnn_num_samples': 1000,
    'bnn_batch_size': 32,
    'num_strata_samples': 5,
    'bnn_training_epochs': 1,
    'bnn_v_prior': 1,
    'bnn_learning_rate': 0.0001,
    'bnn_alpha':0.5,
    'wb_num_epochs':1,
    'wb_learning_rate':0.0001
}
results = {}
run_type = 'full'
#run_type = 'train_model'
create_exp_batch = False
episode_count = 1000 # reduce episode count for demonstration since HiPMDP learns policy quickly
preset_hidden_params = [{'latent_code': 2}]
domain = "frozenlake"
#domain = "frozenlake"
with open('results/frozen_0.7/{}_network_weights_itr_2'.format(domain), 'rb') as f1:
    network_weights1 = pickle.load(f1)
#with open('results/latent2/{}_network_weights'.format(domain), 'rb') as f:
#    network_weights2 = pickle.load(f)
with open('results/frozen_0.7/{}_network_weights_itr_2'.format(domain), 'rb') as f2:
    network_weights2 = pickle.load(f2)
with open('results/frozen_0.7/{}_latent_weights_itr_2'.format(domain), 'rb') as f3:
    latent_weights1 = pickle.load(f3)
#with open('results/latent2/{}_latent_weights'.format(domain), 'rb') as f:
#    latent_weights2 = pickle.load(f)
with open('results/frozen_0.7/{}_latent_weights_itr_2'.format(domain), 'rb') as f4:
    latent_weights2 = pickle.load(f4)
#preset_hidden_params = [{'latent_code': 2}]
results = {}
run_type = 'full'
#run_type = 'train_model'
create_exp_batch = False
episode_count = 1000 # reduce episode count for demonstration since HiPMDP learns policy quickly
preset_hidden_params = [{'latent_code': 1}]

hipmdp1 = HiPMDP(domain,preset_hidden_params,
                 episode_count=episode_count,
                 run_type=run_type,
                 bnn_hidden_layer_size=bnn_hidden_layer_size,
                 bnn_num_hidden_layers=bnn_num_hidden_layers,
                 bnn_network_weights=network_weights1, test_inst=test_inst,
                 eps_min=eps_min, create_exp_batch=create_exp_batch,grid_beta=grid_beta,print_output=True)

hipmdp2 = HiPMDP(domain,preset_hidden_params,
                 episode_count=episode_count,
                 run_type=run_type,
                 bnn_hidden_layer_size=bnn_hidden_layer_size,
                 bnn_num_hidden_layers=bnn_num_hidden_layers,
                 bnn_network_weights=network_weights2, test_inst=test_inst,
                 eps_min=eps_min, create_exp_batch=create_exp_batch,grid_beta=grid_beta,print_output=True)
hipmdp1._HiPMDP__initialize_BNN()
hipmdp2._HiPMDP__initialize_BNN()

weight_set1 = latent_weights1.reshape(latent_weights1.shape[1],)
weight_set2 = latent_weights2.reshape(latent_weights2.shape[1],)
#action = [0,1,2,3]

#state1 = hipmdp1.task.observe()
#state2 = hipmdp2.task.observe()


#state1_t = hipmdp1.network.feed_forward(aug_state1)
#state2_t = hipmdp2.network.feed_forward(aug_state2)



# Define bounds for X and Y coordinates and the set of possible actions
#state_bounds = ((0, 10), (0, 10))
#actions = [0,1,2,3]
lipschitz_constant = 1.5  # Example value; replace with your expected constant





def check_lipschitz_between_bnns(bnn1, bnn2, lipschitz_constant, latent_wight1, latent_weight2):
    """
    Check if the predictions made by two BNNs are bounded by a Lipschitz Constant.

    num_samples: Number of random samples to be drawn.
    lipschitz_constant: Expected Lipschitz constant.
    state_bounds: Tuple indicating the min and max values for X and Y coordinates.
    actions: List of possible actions [e.g., 'up', 'down', 'left', 'right'].

    Returns: Fraction of samples that adhere to the Lipschitz condition between the two BNNs.
    """

    def __decode_state(coordinates):
        """Converts normalized grid coordinates back to the state index."""
        nrow = 7
        ncol = 8
        # Denormalize the coordinates
        row = np.round(coordinates[0] * (nrow - 1))
        col = np.round(coordinates[1] * (ncol - 1))

        # Convert 2D grid coordinates to a single index
        state = row * ncol + col

        return state

    def __decode_state_array(coordinates):
        """Converts normalized grid coordinates back to state indices for multiple coordinates."""
        nrow = 7
        ncol = 8
        coordinates = coordinates.reshape(-1, 2)

        # Assuming coordinates shape is (N, 2), where N is the number of coordinate pairs
        rows = np.round(coordinates[:, 0] * (nrow - 1))
        cols = np.round(coordinates[:, 1] * (ncol - 1))

        # Convert 2D grid coordinates to an array of state indices
        states = rows * ncol + cols

        return states

    action = [0, 1, 2, 3]

    count_adhering = 0
    #aug_state1 = np.hstack([state1, __encode_action(action), weight_set1]).reshape((1, -1))
    #aug_state2 = np.hstack([state2, __encode_action(action), weight_set1]).reshape((1, -1))
    state_list = []
    for s in range(16):
        row, col = bnn1.task.to_m(s)
        letter = bnn1.task.desc[row, col]
        if letter != b'H':  # s is not a Hole
            state_list.append(s)
    W_distances = []
    uncertainties1_e = []
    uncertainties2_e = []
    uncertainties1_a = []
    uncertainties2_a = []
    #print(state_list)
    #for state in state_list:
    # Determine grid size
    num_pairs = len(state_list) * len(action)
    cols = 8  # for example, 4 columns
    rows = (num_pairs // cols) + (1 if num_pairs % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(40, 5 * rows))
    print(state_list)
    i = 0
    mse_values = []

    total_aleatoric_uncertainties = []
    total_epistemic_uncertainties = []
    for a in action:
        for state in state_list:
            print("start", state,a)
            state1 = bnn1.task._NSFrozenLakeV0__encode_state(state)
            #bnn1.task.T(state,a,0)
            #print("check state",state, state1)
            #state1 = bnn1.task._NSFrozenLakeV0__encode_state(state)
            #aug_state1 = np.hstack([state1, __encode_action(a), latent_wight1]).reshape((1, -1))
            aug_state2 = np.hstack([state1, __encode_action(a), latent_weight2]).reshape((1, -1))
            #_, samples1, epistemic_uncertainties1, aleatoric_uncertainties1 = bnn1.network.feed_forward_distribution(aug_state1)
            means, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = bnn2.network.feed_forward_distribution(aug_state2)
            total_aleatoric_uncertainties.append(np.sum(aleatoric_uncertainties2))
            total_epistemic_uncertainties.append(np.sum(epistemic_uncertainties2))
            #print(samples2.reshape(-1, 2))
            state_count = []
            samples2 = samples2[:, 0, :]
            for j in samples2.reshape(-1, 2):
                state_count.append(bnn1.task._NSFrozenLakeV0__decode_state(j,state,a))
            #print(samples2)
            # Example Usage
            from collections import Counter
            print(means)
            #coordinates = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
            #bnn1.task.plot_map_with_coordinates("4x4", samples2, state, a)
            print("check",state, a, bnn1.task._NSFrozenLakeV0__decode_state(means.flatten(),state,a ), np.sum(epistemic_uncertainties2), np.sum(aleatoric_uncertainties2))
            print(bnn1.task.T[state, a, 0])
            print("state list", Counter(state_count))
            state_counts = Counter(state_count)  # Assuming state_count is defined somewhere in your code

            # Calculate total transitions
            total_transitions = sum(state_counts.values())

            # Initialize a list of zeros for all states
            bnn_transitions = [0.0] * 16

            # Fill in the transition probabilities for the states in state_counts
            for s, c in state_counts.items():
                bnn_transitions[s] = c / total_transitions

            true_transitions = bnn1.task.T[state, a, 0]  # Assuming bnn1 is defined somewhere in your code
            mse = np.mean((np.array(bnn_transitions) - np.array(true_transitions))**2)
            mse_values.append(mse)  # Append the MSE to the list
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
            bar_width = 0.35
            index = np.arange(16)
            bar1 = ax.bar(index, true_transitions, bar_width, label='True Dynamics', color='b')
            bar2 = ax.bar(index + bar_width, bnn_transitions, bar_width, label='BNN Predictions', color='r')
            ax.text(0.95, 0, f'Aleatoric Uncertainty: {np.sum(aleatoric_uncertainties2):.2f}', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.text(0.95, 0.05, f'Epistemic Uncertainty: {np.sum(epistemic_uncertainties2):.2f}', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.set_xlabel('States')
            ax.set_ylabel('Transition Probabilities')
            ax.set_title(f'State {state}, Action {a}')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(index)
            ax.legend()

            i += 1
    average_mse = np.mean(mse_values)
    print("mse:", average_mse)
    plt.tight_layout()
    filename = "transitions_comparison.png"
    plt.savefig(filename)
    plt.close(fig)
    print("end")
    print("total total_aleatoric_uncertainties:",total_aleatoric_uncertainties)
    print("average aleatoric_uncertainties:", np.mean(total_aleatoric_uncertainties))
    print("average epistemic_uncertainties:", np.mean(total_epistemic_uncertainties))
    ''''
    state_counts = Counter(state_count)

    # Calculate total transitions
    total_transitions = sum(state_counts.values())

    # Initialize a list of zeros for all states
    bnn_transitions = [0.0] * 16

    # Fill in the transition probabilities for the states in state_counts
    for s, c in state_counts.items():
        bnn_transitions[s] = c / total_transitions
    true_transitions = bnn1.task.T[state, a, 0]
    bar_width = 0.35
    index = np.arange(16)

    fig, ax = plt.subplots()
    bar1 = ax.bar(index, true_transitions, bar_width, label='True Dynamics', color='b')
    bar2 = ax.bar(index + bar_width, bnn_transitions, bar_width, label='BNN Predictions', color='r')

    ax.set_xlabel('States')
    ax.set_ylabel('Transition Probabilities')
    ax.set_title('Comparison of State Transition Probabilities')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(index)
    ax.legend()

    plt.tight_layout()
    filename = f"State_{state}_Action_{a}_transitions.png"  # Reformat filename to avoid issues
    plt.savefig(filename)
    plt.close(fig)
    print("end")
    '''
    '''
    epistemic_distribution = []
    aleatoric_distribution = []
    for state in state_list:
        for a in action:
            state1 = bnn1.task._NSFrozenLakeV0__encode_state(state)
            #print("check state",state, state1)
            #state1 = bnn1.task._NSFrozenLakeV0__encode_state(state)
            #aug_state1 = np.hstack([state1, __encode_action(a), latent_wight1]).reshape((1, -1))
            aug_state2 = np.hstack([state1, __encode_action(a), latent_weight2]).reshape((1, -1))
            #_, samples1, epistemic_uncertainties1, aleatoric_uncertainties1 = bnn1.network.feed_forward_distribution(aug_state1)
            means, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = bnn2.network.feed_forward_distribution(aug_state2)
            epistemic_distribution.append(np.sum(epistemic_uncertainties2))
            aleatoric_distribution.append(np.sum(aleatoric_uncertainties2))
            #print(samples2.reshape(-1, 2))
    epistemic_distribution = np.array(epistemic_distribution)
    aleatoric_distribution = np.array(aleatoric_distribution)
    print(epistemic_distribution.shape)
    # Calculate mean values
    epistemic_mean = np.mean(epistemic_distribution)
    aleatoric_mean = np.mean(aleatoric_distribution)

    # Plot and save for epistemic distribution
    plt.figure(figsize=(6, 6))
    plt.hist(epistemic_distribution, bins=30, alpha=0.7, color='blue', label='Epistemic Distribution')
    plt.axvline(epistemic_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {epistemic_mean:.2f}')
    plt.title("Epistemic Uncertainty Distribution")
    plt.legend()
    plt.savefig("Epistemic_Uncertainty.png", dpi=300)
    plt.close()

    # Plot and save for aleatoric distribution
    plt.figure(figsize=(6, 6))
    plt.hist(aleatoric_distribution, bins=30, alpha=0.7, color='green', label='Aleatoric Distribution')
    plt.axvline(aleatoric_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {aleatoric_mean:.2f}')
    plt.title("Aleatoric Uncertainty Distribution")
    plt.legend()
    plt.savefig("Aleatoric_Uncertainty.png", dpi=300)
    plt.close()



    for state in state_list:
        for a in action:
            state1 = bnn1.task._NSBridgeV0__encode_state(state)
            aug_state1 = np.hstack([state1, __encode_action(a), latent_wight1]).reshape((1, -1))
            aug_state2 = np.hstack([state1, __encode_action(a), latent_weight2]).reshape((1, -1))
            _, samples1, epistemic_uncertainties1, aleatoric_uncertainties1 = bnn1.network.feed_forward_distribution(aug_state1)
            _, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = bnn2.network.feed_forward_distribution(aug_state2)
            #uncertainty1 = np.sum(np.std(output_bnn1, axis=0))
            #print(aleatoric_uncertainties1)
            uncertainties1_e.append(np.sum(epistemic_uncertainties1))
            uncertainties1_a.append(np.sum(aleatoric_uncertainties1))
            #print("uncertainty1",uncertainty1)
            #print("uncertainty1",output_bnn1)
            #uncertainty2 = np.sum(np.std(output_bnn2, axis=0))
            #print("uncertainty2", uncertainty2)
            uncertainties2_e.append(np.sum(epistemic_uncertainties2))
            uncertainties2_a.append(np.sum(aleatoric_uncertainties2))
            #decoded_bnn1 = __decode_state_array(output_bnn1)
            #decoded_bnn2 = __decode_state_array(output_bnn2)
            #print("uncertainty1", uncertainty1)
            #print("uncertainty2", uncertainty2)
            # Flatten your data to shape (50, 2)
            samples1 = samples1[:, 0, :]
            samples2 = samples2[:, 0, :]
            #print(decoded_bnn1)
            #print(decoded_bnn2)
            #samples1 = decoded_bnn1
            #samples2 = decoded_bnn2
            # Compute pairwise distance matrix between the two sets of samples
            M = ot.dist(samples1, samples2)

            # Compute the 2-Wasserstein distance
            # Assuming both sets of samples represent uniform distributions over the set
            a, b = np.ones((50,)) / 50, np.ones((50,)) / 50
            emd = ot.emd2(a, b, M)

            print("2-Wasserstein Distance:", emd)
            #distance = wasserstein_distance(output_bnn1, output_bnn2)
            #print(distance)
            # = np.linalg.norm(output_bnn1 - output_bnn2)  # Compute the norm for the difference
            #if output_diff <= lipschitz_constant:
            #    count_adhering += 1
            W_distances.append(emd)
    ###############################################
    plt.figure()

    # Calculate the mean value
    mean_val = np.mean(W_distances)
    # Plotting the distribution
    sns.histplot(W_distances, bins=20,
                 kde=True)  # 'bins' determines the number of bins in the histogram, 'kde' adds a density plot

    # Add a vertical line at the mean value
    plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.4f}")

    plt.title("Distribution of EMD values")
    plt.xlabel("EMD Value")
    plt.ylabel("Frequency")
    plt.legend()  # This will display the label for the mean line

    # Save the figure before showing it
    plt.savefig("emd_distribution.png", dpi=300)  # dpi (dots per inch) controls the resolution
    ###################################################3
    plt.figure()

    # Calculate the mean value
    mean_val = np.mean(uncertainties1_e)
    # Plotting the distribution
    sns.histplot(uncertainties1_e, bins=20,
                 kde=True)  # 'bins' determines the number of bins in the histogram, 'kde' adds a density plot

    # Add a vertical line at the mean value
    plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.4f}")

    plt.title("epistemic_uncertainties1")
    plt.xlabel("EMD Value")
    plt.ylabel("Frequency")
    plt.legend()  # This will display the label for the mean line

    # Save the figure before showing it
    plt.savefig("epistemic_uncertainties1.png", dpi=300)  # dpi (dots per inch) controls the resolution
    ###############################################################
    plt.figure()

    # Calculate the mean value
    mean_val = np.mean(uncertainties1_a)

    # Plotting the distribution
    sns.histplot(uncertainties1_a, bins=20,
                 kde=True)  # 'bins' determines the number of bins in the histogram, 'kde' adds a density plot

    # Add a vertical line at the mean value
    plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.4f}")

    plt.title("uncertainties1_aleatoric")
    plt.xlabel("EMD Value")
    plt.ylabel("Frequency")
    plt.legend()  # This will display the label for the mean line

    # Save the figure before showing it
    plt.savefig("uncertainties1_aleatoric.png", dpi=300)  # dpi (dots per inch) controls the resolution
    #num_samples = len(state_list) * len(action)
    #print(count_adhering, num_samples)
    #return count_adhering / num_samples
    ###############################################################
    plt.figure()

    # Calculate the mean value
    mean_val = np.mean(uncertainties2_a)

    # Plotting the distribution
    sns.histplot(uncertainties2_a, bins=20,
                 kde=True)  # 'bins' determines the number of bins in the histogram, 'kde' adds a density plot

    # Add a vertical line at the mean value
    plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.4f}")

    plt.title("uncertainties2_aleatoric")
    plt.xlabel("EMD Value")
    plt.ylabel("Frequency")
    plt.legend()  # This will display the label for the mean line

    # Save the figure before showing it
    plt.savefig("uncertainties2_aleatoric.png", dpi=300)  # dpi (dots per inch) controls the resolution

    ###############################################################
    plt.figure()

    # Calculate the mean value
    mean_val = np.mean(uncertainties2_e)

    # Plotting the distribution
    sns.histplot(uncertainties2_e, bins=20,
                 kde=True)  # 'bins' determines the number of bins in the histogram, 'kde' adds a density plot

    # Add a vertical line at the mean value
    plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.4f}")

    plt.title("uncertainties2_epistemic")
    plt.xlabel("EMD Value")
    plt.ylabel("Frequency")
    plt.legend()  # This will display the label for the mean line

    # Save the figure before showing it
    plt.savefig("uncertainties2_epistemic.png", dpi=300)  # dpi (dots per inch) controls the resolution
'''

# Check the Lipschitz property between the two BNNs and print the fraction of adhering samples
fraction_adhering = check_lipschitz_between_bnns(hipmdp1, hipmdp2, lipschitz_constant, weight_set1, weight_set2)
#print(f"Fraction of samples adhering to Lipschitz property between the two BNNs: {fraction_adhering:.2f}")
