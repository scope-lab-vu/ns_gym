import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import ns_gym.base as base
import numpy as np
from ns_gym.benchmark_algorithms.algo_utils import observation_type_checker, reward_type_checker
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torch.nn.functional as F
import os

"""
Simple Deep Q Network (DQN) algorithm for benchmarking: See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


Deep Q Network will train on a stationary environment, but will be tested on a non-stationary environment.


Do you have an idea of how much would be a reasonable amount of resources to request? One thing I need to do is train a simple DDQN model for at least 7 different environments, so a GPU would be useful. 
"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def network_input_checker(x, state_size):
    """Make sure the input to the neural network is in the correct format
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if isinstance(x, base.Observation):
        x = x.state

    s = x 
    if type(s) == int:
        arr = torch.zeros(state_size)
        arr[s] = 1
        arr = arr.unsqueeze(0)
        s = arr

    if not isinstance(s, torch.Tensor):
        s = torch.Tensor(x)

    s = s.to(device)
    return s


class ReplayBuffer(object):
    """Replay buffer to store and sample experience tuples.
    """
    def __init__(self, capacity,state_size, action_size) -> None:
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.state_size = state_size
        self.action_size = action_size

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def add(self, state, action, reward, next_state,done):

        if isinstance(state, base.Observation):
            state = state.state
        if isinstance(next_state, base.Reward):
            reward = reward.reward

        state = network_input_checker(state,state_size=self.state_size)
        next_state = network_input_checker(next_state,state_size=self.state_size)

        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
 
    def sample(self,batch_size):
        experiences = random.sample(self.buffer, k=batch_size)

        states = np.stack([e.state.cpu() for e in experiences if e is not None])



        states = torch.from_numpy(np.vstack([e.state.cpu() for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state.cpu() for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q network, maps state to action values.
    """

    def __init__(self, state_size, action_size, num_layers, num_hidden_units, seed): 
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.activation = nn.ReLU()


        arch = []
        for i in range(num_layers):
            if i == 0:
                arch.append(nn.Linear(state_size,num_hidden_units))
                arch.append(self.activation)
            elif i == num_layers - 1:
                arch.append(nn.Linear(num_hidden_units,action_size))
            else:
                arch.append(nn.Linear(num_hidden_units,num_hidden_units))
                arch.append(self.activation)

        self.layers = nn.Sequential(*arch)

    def forward(self, state):
        # state = self.type_checker(state)
        # state = network_input_checker(state, self.state_size)

        return self.layers(state)
    
    def type_checker(self, x):
        if isinstance(x, base.Observation):
            return x.state
        else:
            return x
    

class DQNAgent(base.Agent):
    def __init__(self,
                 state_size, 
                 action_size, 
                 seed,
                 model  = None,
                 model_path = None,
                 buffer_size=int(1e5),
                 batch_size=64,
                 gamma=0.99,
                 lr=0.001,
                 update_every=4,
                 do_update=False) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


        if model:
            self.q_network_local = model
            self.q_network_local = self.q_network_local.to(self.device)
            self.q_network_target = model
            self.q_network_target = self.q_network_target.to(self.device)
            if model_path:
                model_weights = torch.load(model_path,map_location=self.device)
                self.q_network_local.load_state_dict(model_weights)
                self.q_network_target.load_state_dict(model_weights)
        else:
            self.q_network_local = DQN(state_size, action_size, num_layers=3, num_hidden_units=64, seed=seed)
            self.q_network_target = DQN(state_size, action_size, num_layers=3, num_hidden_units=64, seed=seed)

        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=lr,weight_decay=1e-5)
        self.memory = ReplayBuffer(buffer_size, state_size, action_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0
    

        self.do_update = do_update
        self.q_network_local = self.q_network_local.to(self.device)
        self.q_network_target = self.q_network_target.to(self.device)


    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences, self.gamma)

    def search(self, state, eps=0.):
        state = network_input_checker(state,state_size=self.state_size)
        self.q_network_local.eval()
        with torch.no_grad():
            s = network_input_checker(state, self.state_size)
            action_values = self.q_network_local(s)
        self.q_network_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()),action_values.cpu().data.numpy().ravel()
        else:
            return random.choice(np.arange(self.action_size)),action_values.cpu().data.numpy()
    
    def act(self, state,eps):
        best_action, _ = self.search(state,eps=eps)
        return best_action
    

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        next_states = network_input_checker(next_states,state_size=self.state_size)
        states = network_input_checker(states,state_size=self.state_size)

        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.q_network_local(states).gather(1,actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_network_local, self.q_network_target, tau=1e-3)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
             target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def do_gradient_updates(state,env, agent, time_budget,eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    start = time.time()
    eps = eps_start
    while time.time() - start < time_budget and agent.do_update:
        while True:
            action, values = agent.act(state, eps=0)
            next_state, reward, done, truncated,_ = env.step(action)
            next_state, reward = observation_type_checker(next_state), reward_type_checker(reward)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done or truncated or time.time() - start > time_budget:
                state,_ = env.reset(seed = random.randint(0,100000))
                if isinstance(state, base.Observation):
                    state = state.state
                break
        eps = max(eps_end, eps_decay * eps)



def train_ddqn(env, model, n_episodes=1000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.999, seed=0):
    """DDQN Training Loop
    """
    agent = DQNAgent(model.state_size, model.action_size, seed=seed, model=model)   
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    best_score = -math.inf
    for i_episode in range(1, n_episodes + 1):
        state,_ = env.reset(seed = random.randint(0,100000))
        if isinstance(state, base.Observation):
            state = state.state
        score = 0
        t = 0
        while True:
            action, values = agent.search(state, eps)
            next_state, reward, done, truncated,_ = env.step(action)
            next_state, reward = observation_type_checker(next_state), reward_type_checker(reward)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or truncated or t>max_t:
                break
            t += 1
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        if  np.mean(scores_window) > best_score:
            best_score = np.mean(scores_window)
            current_dir = os.path.dirname(__file__) 
            os.makedirs(os.path.join(current_dir,'DDQN_models'), exist_ok=True)
            saved_model_dir = os.path.join(current_dir,'DDQN_models')
            print("Model Saved")
            print(f'\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.q_network_local.state_dict(), os.path.join(saved_model_dir,"Bridge_DDQN_2Layers_128_2.pth"))

    return agent, scores




if __name__ == "__main__":
    pass



