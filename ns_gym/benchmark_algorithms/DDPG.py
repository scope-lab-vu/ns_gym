import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from ns_gym import base

import math


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, action_dim)
        self.activation = nn.ReLU()  # Hidden layers activation

    def forward(self, state):
        """
        Forward pass through the actor network.

        Args:
            state: Tensor of shape (batch_size, state_dim).

        Returns:
            action: Tensor of shape (batch_size, action_dim).
        """

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32) 

        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        action = self.fc_out(x)
        return action
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_size=256):
        super(Critic, self).__init__()
        # First layer processes both state and action
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()  # Non-linear activation

    def forward(self, state, action):
        """
        Forward pass through the Critic network.

        Args:
            state: Tensor of shape (batch_size, state_dim).
            action: Tensor of shape (batch_size, action_dim).

        Returns:
            Q-value: Tensor of shape (batch_size, 1).
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)

        # Forward pass through layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation in the output layer
        return x

class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        """
        A replay buffer that is compatible with PyTorch DataLoader.

        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # Tracks the position to overwrite old data

    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.

        Args:
            state: The current state (torch tensor or numpy array).
            action: The action taken.
            reward: The reward received.
            next_state: The next state (torch tensor or numpy array).
            done: Boolean indicating if the episode ended.
        """
        # Convert all inputs to tensors
        transition = (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)

    def __getitem__(self, idx):
        """
        Retrieve a single transition from the buffer.

        Args:
            idx (int): Index of the transition.

        Returns:
            tuple: A single transition (state, action, reward, next_state, done).
        """
        return self.buffer[idx]


class DDPG(base.Agent):
    """Deep Deterministic Policy Gradient (DDPG) algorithm.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_size (int): Number of hidden units in each layer of the networks.
        lr_actor (float): Learning rate for the actor network.
        lr_critic (float): Learning rate for the critic network.

    
    Warning:
        This implementation works though the StableBaselines3 implementation is likely better optimized.
    """

    def __init__(self,state_dim=8,action_dim=2,hidden_size=256,lr_actor=0.001,lr_critic=0.001):
        
        ####### Initialize the Actor and Critic Networks ########
        self.actor = Actor(state_dim,action_dim)

        actor_weights = self.actor.state_dict()

        # make sure actor target have the same initial weights
        self.actor_target = Actor(state_dim,action_dim)
        self.actor_target.load_state_dict(actor_weights)

        self.best_score = -math.inf

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=lr_actor)

        self.critic = Critic(state_dim,action_dim)

        # make sure critic target have the same initial weights
        self.critic_target = Critic(state_dim,action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=lr_critic)


        ######### Initialize the Replay Buffer #########
        self.replay_buffer = ReplayBuffer(capacity=1000000)


    def train(self, env, num_episodes=10000, batch_size=64, gamma=0.99, tau=0.005,warmup_episodes=300,save_path="models/"):   
        
        self.warmup(env,warmup_episodes) # Warmup the replay buffer

        ep_reward_list = []
        self.actor.train()
        self.critic.train()

        initial_noise_scale = 0.1
        decay_rate = 0.99
        min_noise_scale = 0.01

        ep = 0
        self.best_score = -math.inf

        while True:
 
            state,info = env.reset()
            t = 0
            done = False 
            truncated = False

            reward_list = []
            critic_loss_list = []
            actor_loss_list = []

            while not done and not truncated: 

                # take an action in the environment

                # noise_scale = max(initial_noise_scale * (decay_rate ** ep), min_noise_scale)

                # if ep < 600:
                #     noise = torch.randn(2) * 0.15
                # else:
                #     noise = torch.randn(2) * 0.05

                # noise = torch.randn(2) * 0.15
                action = self.actor(torch.tensor(state)) + noise
                action = torch.clamp(action, -1, 1)  # Clip action to [-1, 1]

                with torch.no_grad():
                    action = action.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy
                next_state, reward, done, truncated, info = env.step(action)


                # store the reward
                reward_list.append(reward)

                # add the transition to the replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state


                # sample a batch from the replay buffer
                dl = DataLoader(self.replay_buffer, batch_size=batch_size, shuffle=True)
                sample = next(iter(dl))
                states, actions, rewards, next_states, dones = sample

                # update the actor and critic networks
                critic_loss,actor_loss = self.update(states, actions, rewards, next_states, dones, gamma, tau)

                critic_loss_list.append(critic_loss)
                actor_loss_list.append(actor_loss)  
                t += 1
            
            ep += 1
            ep_reward_list.append(sum(reward_list))

            if ep % 100 == 0:
                print(f"Episode: {ep}, Reward: {sum(reward_list)}, Critic Loss: {sum(critic_loss_list)/len(critic_loss_list)}, Actor Loss: {sum(actor_loss_list)/len(actor_loss_list)}")
                print("ave reward: ", sum(ep_reward_list[-100:])/100)


            if len(ep_reward_list) > 100:
                avg_reward = sum(ep_reward_list[-100:])/100
                if avg_reward > self.best_score:
                    self.best_score = avg_reward
                    torch.save(self.actor.state_dict(), save_path+"actor.pth")
                    torch.save(self.critic.state_dict(), save_path+"critic.pth")


        return self.best_score
                

    def update(self, states, actions, rewards, next_states, dones, gamma=0.99, tau=0.001):
        """
        Update the actor and critic networks for one training step in DDPG.

        Args:
            states: Batch of current states.
            actions: Batch of actions taken.
            rewards: Batch of rewards received.
            next_states: Batch of next states.
            dones: Batch of done flags (indicating episode termination).
            gamma: Discount factor.
            tau: Target network soft update parameter.
        """
        # Convert dones to a mask for termination
        dones = dones.float()

        # 1. Update Critic Network
        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, next_actions)
            target_q_values = target_q_values.squeeze(1)
            y = rewards + gamma * target_q_values * (1 - dones)  # Mask out terminal states

        # Compute critic loss
        current_q_values = self.critic(states, actions)
        current_q_values = current_q_values.squeeze(1)


        critic_loss = nn.MSELoss()(current_q_values, y)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Update Actor Network
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Soft Update Target Networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()



    def warmup(self,env,warmup_episodes=300):
        for i in range(warmup_episodes):
            state,info = env.reset()
            done = False
            truncated = False

            t = 0

            print(f"Warmup Episode: {i}")

            while not (done or truncated):
                
                action = self.actor(torch.tensor(state)) + torch.randn(2) * 0.15 # Add noise to the action
                
                with torch.no_grad():
                    action = action.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

                next_state, reward, done, truncated, info = env.step(action)
                # if truncated:
                #     print("Truncated")

                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                t += 1

                # if done or truncated:
                #     state,info = env.reset()

        
        print("Warmup Done")


    def run_eval_episode(self,env,T=100,visualize=False):

        self.actor.eval()

        reward_list = []

        state,info = env.reset()
        done = False
        truncated = False
        t = 0

        while not done and not truncated:
            action = self.actor(state)

            with torch.no_grad():
                action = action.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

            next_state, reward, done, truncated, info = env.step(action)
            if visualize:
                env.render()
            if truncated:
                print("Truncated")
            reward_list.append(reward)
            state = next_state


        return sum(reward_list), reward_list
    

    def act(self,observation,clip=False):

        if clip:
            return torch.clamp(self.actor(observation),-1,1)
        
        return self.actor(observation)


if __name__ == '__main__':
    import gymnasium as gym


    env = gym.make("LunarLanderContinuous-v3")
