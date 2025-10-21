import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
from ns_gym.benchmark_algorithms.AlphaZero.AlphaZeroMCTSv4 import MCTS 
import torch.nn.functional as F
import ns_gym.base as base
import os
import ns_gym as nsg
from collections import deque
########### ALPHA ZERO NETWORK ################
class AlphaZeroNetwork(nn.Module):
    """
    Overview:
        This is a simple MLP that predicts the policy and value of a particalar state

        Args:
            action_space_dim (int): Size of the action space
            observation_space_dim (int): Size of the observation space
            lr (float): learning rate
            n_hidden_layers (int): Number of hidden layers
            n_hidden_units (int): Number of units in each hidden layer
            activation (str, optional): Activation . Defaults to 'relu'.
    """

    def __init__(self,action_space_dim, observation_space_dim,n_hidden_layers, n_hidden_units,activation='relu'):

        super().__init__()

        self.action_dim = action_space_dim
        self.obs_dim = observation_space_dim

        if activation == 'relu':
            self.activation = torch.nn.ReLU()

        # self.layers = nn.ModuleList()
        layers = []
        input_layer = nn.Linear(self.obs_dim,n_hidden_units)

        layers.append(input_layer)
        layers.append(self.activation)

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_units,n_hidden_units))
            layers.append(self.activation)

        self.layers = nn.Sequential(*layers)
        self.value_head = nn.Linear(n_hidden_units,1)
        self.policy_head = nn.Linear(n_hidden_units,self.action_dim)

        # self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        
    def forward(self,obs):
        """
        Overview:
            A single forward pass of the observations from the environment

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: Tuple containing the policy and value of the state
        """
        if not isinstance(obs,torch.Tensor):
             obs = torch.Tensor(obs)
        x = self.layers(obs)
        v = self.value_head(x)
        pi = self.policy_head(x)
        return F.softmax(pi,dim=-1), v
    
    def input_check(self,obs):
        if type(obs) == int:
            arr = torch.zeros(self.env.observation_space.n)
            arr[obs] = 1
            arr = arr.unsqueeze(0)
            obs = arr
        return obs

########### REPLAY BUFFER ################
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.idx = 0

    def add(self, state,value,pi):

        experience = (state,value,pi)
        if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
        else:
                self.buffer[self.idx] = experience
                self.idx = (self.idx + 1) % self.max_size

    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        if buffer_size < batch_size:
            batch_size = buffer_size  # Sample all available elements if not enough in the buffer

        indices = np.random.choice(buffer_size, batch_size, replace=False)
        states, values, pis = zip(*[self.buffer[idx] for idx in indices])

        states = np.stack(states)
        pis = np.stack(pis)

        states = torch.from_numpy(states).float()
        values = torch.from_numpy(np.array(values)).float()
        pis = torch.from_numpy(pis).float()

        return states, values, pis
    
    def all_samples(self):
        states, values, pis = zip(*self.buffer)

        states = np.stack(states)
        pis = np.stack(pis)

        states = torch.from_numpy(states).float()
        values = torch.from_numpy(np.array(values)).float()
        pis = torch.from_numpy(pis).float()
        
        return states, values, pis



############ TRAINING FUNCTION ############
def train_model(buffer,model,lr,n_epochs,num_episodes,batch_size,weight_decay,ep,writer):
    """
    Overview:
        Train the model on the trajectory

    """

    optimizer = optim.Adam(model.parameters(),lr = lr,weight_decay=weight_decay)
    criterion_V = F.mse_loss
    criterion_pi = F.cross_entropy
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.train()
    batch_num = 0 

    # states,value,pi = buffer.sample(batch_size)
    states,value,pi = buffer.all_samples()

    states = states.to(device)
    s_size = states.size()
    value = value.to(device)
    value_size = value.size()
    pi = pi.to(device)
    pi_size = pi.size()
        
        
    for epoch in range(n_epochs):

        pred_pi,pred_v = model(states)
        optimizer.zero_grad()
        pred_v = pred_v.squeeze()
        value = value.squeeze()

  
        #### DEBUG ####
        pred_pi = pred_pi.squeeze(1)
        assert pred_v.shape == value.shape
        lv = criterion_V(pred_v,value)

        assert pred_pi.shape == pi.shape
        lpi = criterion_pi(pred_pi, pi)

        #Total loss
        loss = lv + lpi

        # Backpropagation
        loss.backward()
        optimizer.step()

        writer.add_scalar(f"Training Loss ep : {ep}", loss.item(), epoch + batch_num*n_epochs)





########## TYPE CHECKER ####################

def type_checker(state,reward):
    """Type checker to ensure compatibility between ns_gym and Gymnasium environments.

    Args:
        state (Union[int, np.ndarray,dict]): The observation, which may be an dictionary.
        reward (Union[float, ns_gym.base.Reward]): The reward, which may be an instance of ns_gym.base.Reward.
    """
    if isinstance(state, dict) and 'state' in state:
        state = state['state']
    if isinstance(reward,base.Reward):
        reward = reward.reward
    return state,reward


########## ALPHA ZERO AGENT ################

class AlphaZeroAgent:
    def __init__(self,
                action_space_dim:int,
                observation_space_dim:int,
                n_hidden_layers,
                n_hidden_units,
                gamma,
                c,
                num_mcts_simulations,
                max_mcts_search_depth,
                model_checkpoint_path = None,
                model:AlphaZeroNetwork = AlphaZeroNetwork,
                alpha = 1.0,
                epsilon = 0.0
                ) -> None:
        """
        Arguments:
            Env: gym.Env
            model: AlphaZeroNetwork : Neurla network,f(obs) the outputs v and phi
            mcts: MCTS
            lr : float : learning rate for NN
            n_hidden_layers: int: number of hidden layers in model
            n_hidden_units: int: number of units in each layer of model
            n_episodes: int: Number of training episodes (number ot times we do mcts + model training)
            max_episode_len : int: The maximum number of environment steps before we terminate
        """
        self.model = model(action_space_dim, observation_space_dim,n_hidden_layers, n_hidden_units,activation='relu')


        ######## SET DEVICE ########
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        if model_checkpoint_path:
            checkpoint = torch.load(model_checkpoint_path,map_location=self.device)
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.mcts = MCTS # just the MCTS object ...
        self.gamma = gamma
        self.c = c
        self.num_mcts_simulations = num_mcts_simulations
        self.max_mcts_search_depth = max_mcts_search_depth
        self.alpha = alpha
        self.epsilon = epsilon


    def train(self,
            env,
            n_episodes,
            max_episode_len,
            lr,
            batch_size, 
            n_epochs,
            experiment_name,
            eval_window_size = 100,
            weight_decay = 1e-4,
            temp_start = 2,
            temp_end = 0.8,
            temp_decay = 0.95
        ):
        """Train the AlphaZero agent

        Args:
            env (gym.Env): The environment to train on
            n_episodes (int): Number of training episodes
            max_episode_len (int): Maximum number of steps per episode
            lr (float): Learning rate for the neural network
            batch_size (int): Batch size for training
            n_epochs (int): Number of epochs per training iteration
            experiment_name (str): Name for saving models and logs
            eval_window_size (int, optional): Size of the evaluation window. Defaults to 100.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-4.
            temp_start (float, optional): Starting temperature for exploration. Defaults to 2.
            temp_end (float, optional): Ending temperature for exploration. Defaults to 0.8.
            temp_decay (float, optional): Decay rate for temperature. Defaults to 0.95.

        Returns:
            List[float]: List of episode returns

        """

        self.Env = env
        episode_returns = []
        t_total = 0 # total number of steps
        R_best = -np.inf

        writer = SummaryWriter("runs/"+experiment_name)

        seeds = random.sample(range(1000000),n_episodes)
        eval_window = deque(maxlen = eval_window_size) #Keep a window of the last 100 episodes 
        temp = temp_start

        for ep, seed in enumerate(seeds):
            print("Staring New episode: ", ep)
            obs,_ = self.Env.reset(seed = seed) #init obs # TODO: add random seed
            R = 0.0
            a_store = []
            buffer = ReplayBuffer(max_size= max_episode_len)

            self.model.eval()
            for t in range(max_episode_len):
                #MCTS step
                mcts = self.mcts(self.Env,state=obs,model=self.model,d=self.max_mcts_search_depth,m=self.num_mcts_simulations,c=self.c,gamma=self.gamma,alpha=self.alpha,epsilon=self.epsilon) 
                mcts.search()
                state,pi,V,_ = mcts.return_results(temp = temp)
                #### STORE THE STATE,PI,V IN THE BUFFER ####
                buffer.add(state,V,pi)

                #### TAKE A STEP IN ENV  ####
                a = np.random.choice(len(pi), p = pi) # Is this right? 
                a_store.append(a)
                
                obs, r, done, truncated,_ = self.Env.step(a)
                obs, r = type_checker(obs,r)

                R+=r

                if done:
                    print("DONE ---- Episode Reward: ", R)
                    break

                elif truncated:
                    print("Truncated ---- Episode Reward: ", R)
                    print("Episode Length: ", t)
                    print("A store: ", a_store) 
                    break
                
            temp = max(temp*temp_decay,temp_end)
            eval_window.append(R)

            if len(eval_window) == eval_window_size:            
                mean_reward_over_last_100_ep = np.mean(eval_window)

                if mean_reward_over_last_100_ep  > R_best:
                    print("Mean reward: ", mean_reward_over_last_100_ep)
                    mod_name = experiment_name+'_best_model_checkpoint.pth'
                    torch.save(self.model.state_dict(), mod_name)
                    R_best = mean_reward_over_last_100_ep

            if ep % 10 == 0:
                torch.save(self.model.state_dict(),experiment_name+f'ep_{ep}_model_checkpoint.pth')

            writer.add_scalar("Episode Rewards", R, ep)

            episode_returns.append(R)
            print("Training Model")
            print(f"Iter {ep}")
            # while buffer.buffer:
            train_model(buffer = buffer,model =self.model,lr = lr,n_epochs = n_epochs,num_episodes = n_episodes,batch_size = batch_size,weight_decay=weight_decay,ep=ep,writer=writer)
        writer.close()
        torch.save(self.model.state_dict(),experiment_name+'_final_model_checkpoint.pth')
        return episode_returns
    
    def act(self,obs,env,temp=1):
        """Use the trained model to select an action

        Args:
            obs (Union[np.array,int,dict]): observation from the environment
            env (gym.Env): The current environment.

        Returns:
            best_action (int)
        """

        obs,_ = nsg.utils.type_mismatch_checker(observation=obs,reward=None)

        self.model.eval()
        mcts = self.mcts(env, state=obs, model=self.model, d=self.max_mcts_search_depth, m=self.num_mcts_simulations, c=self.c, gamma=self.gamma,alpha=self.alpha,epsilon=self.epsilon)
        mcts.search()
        state,pi,V,best_action= mcts.return_results(temp = temp)
        # a = np.random.choice(len(pi), p = pi) 
        return best_action

                 

def train_alpha_zero(config_file_path):
    """Train the AlphaZero agent using yaml configuration file

    Args:
        config_file_path (str): Path the alphazero config yaml file
    """
    import yaml

    with open(config_file_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    ########## SET UP TRAINING ENVIRONMENT ##########

    env_config = config["gym_env"]
    wrapper_config = config["wrapper"]
    agent_config = config["agent"]


    env = gym.make(env_config["name"],is_slippery = env_config["is_slippery"],render_mode = env_config["render_mode"],max_episode_steps=env_config["max_episode_steps"])
    fl_scheduler = nsg.schedulers.ContinuousScheduler()
    fl_updateFn = nsg.update_functions.DistributionNoUpdate(fl_scheduler)

    param = {"P":fl_updateFn}

    env = nsg.wrappers.NSFrozenLakeWrapper(env, 
                                           param,
                                           change_notification = config["ns_frozen_lake_wrapper"]["change_notification"], 
                                           delta_change_notification = config["ns_frozen_lake_wrapper"]["delta_change_notification"], 
                                           in_sim_change = config["ns_frozen_lake_wrapper"]["in_sim_change"], 
                                           initial_prob_dist=config["ns_frozen_lake_wrapper"]["initial_prob_dist"],
                                           modified_rewards={"H":config["ns_frozen_lake_wrapper"]["modified_rewards"]["H"],
                                                             "G":config["ns_frozen_lake_wrapper"]["modified_rewards"]["G"],
                                                             "F":config["ns_frozen_lake_wrapper"]["modified_rewards"]["F"],
                                                             "S":config["ns_frozen_lake_wrapper"]["modified_rewards"]["S"]})
    
    ############ SET UP ALPHA ZERO AGENT ############



    alpha_config = config["alphazero_agent"]
    alphazero_agent = AlphaZeroAgent(
                                     action_space_dim=alpha_config["action_space_dim"],
                                     observation_space_dim=alpha_config["observation_space_dim"],
                                     model=AlphaZeroNetwork,
                                     lr=alpha_config["lr"],
                                     n_hidden_layers=alpha_config["n_hidden_layers"],
                                     n_hidden_units=alpha_config["n_hidden_units"],
                                     gamma=alpha_config["gamma"],
                                     c=alpha_config["c"],
                                     num_mcts_simulations=alpha_config["num_mcts_simulations"],
                                     max_mcts_search_depth=alpha_config["max_mcts_search_depth"],
                                     model_checkpoint_path = None)
    

    ############ TRAINING ############
    experiment_dir = config["experiment_dir"]
    os.makedirs(os.path.join(experiment_dir,"AlphaZeroFrozenLake"),exist_ok=True)
    experiment_name  = os.path.join(experiment_dir,"AlphaZeroFrozenLake",config["experiment_name"])

    training_config = config["training"]
    episode_returns = alphazero_agent.train(env=env,
        n_episodes=training_config["n_episodes"],
                                            max_episode_len=training_config["max_episode_len"],
                                            lr=training_config["lr"],
                                            batch_size=training_config["batch_size"],
                                            n_epochs=training_config["n_epochs"],
                                            experiment_name=experiment_name)




    

if __name__ == "__main__":
    pass