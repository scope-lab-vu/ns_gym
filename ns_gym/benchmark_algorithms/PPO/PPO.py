import ns_gym
import ns_gym.base as base
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time

def initialize_buffers(state_dim, action_dim, max_steps):
    """Initialize buffers for states, actions, values, rewards, and log probabilities."""
    buffers = {
        "states": np.zeros((max_steps, state_dim), dtype=np.float32),
        "actions": np.zeros((max_steps, action_dim), dtype=np.float32),
        "values": np.zeros((max_steps,), dtype=np.float32),
        "rewards": np.zeros((max_steps,), dtype=np.float32),
        "log_probs": np.zeros((max_steps,), dtype=np.float32),
    }
    return buffers

def compute_discounted_returns(rewards, gamma, last_value):
    """Compute discounted returns."""
    returns = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            returns[t] = rewards[t] + gamma * last_value
        else:
            returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns

def compute_gae(rewards, values, gamma, lamb, last_value):
    """Compute Generalized Advantage Estimation (GAE)."""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lamb * last_gae
    return advantages + values

def run_environment(env, buffers, policy_net, value_net, state_dim, action_dim, max_steps, gamma, lamb, device):
    """
    Run an episode in the environment, collecting states, actions, rewards, and other data.
    """
    state = env.reset()[0]
    episode_length = max_steps

    for step in range(max_steps):
        state_tensor = torch.tensor(state[None, :], dtype=torch.float32, device=device)
        action, log_prob = policy_net(state_tensor)
        value = value_net(state_tensor)

        # Store data in buffers
        buffers["states"][step] = state
        buffers["actions"][step] = action.cpu().numpy()[0]
        buffers["log_probs"][step] = log_prob.cpu().numpy()
        buffers["values"][step] = value.cpu().numpy()

        # Take a step in the environment
        state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
        buffers["rewards"][step] = reward
        if terminated or truncated:
            episode_length = step + 1
            break

    # Compute the returns
    last_value = value_net(torch.tensor(state[None, :], dtype=torch.float32, device=device)).cpu().numpy()
    returns = compute_discounted_returns(buffers["rewards"][:episode_length], gamma, last_value)

    # Uncomment the line below to use GAE instead of discounted returns
    # returns = compute_gae(buffers["rewards"][:episode_length], buffers["values"][:episode_length], gamma, lamb, last_value)

    return (
        buffers["states"][:episode_length],
        buffers["actions"][:episode_length],
        buffers["log_probs"][:episode_length],
        buffers["values"][:episode_length],
        returns,
        buffers["rewards"][:episode_length],
    )


class Dist(torch.distributions.Normal):
    """Distribution exploration
    """
    def log_probs(self, x):
        return super().log_prob(x).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class PPOActor(nn.Module):
    """Actor network for policy approximation.

    Outputs mean and standard deviation of the action distribution. A simple MLP.

    Args:
        s_dim: State dimension.
        a_dim: Action dimension.
        hidden_size: Number of hidden units in each layer.
    """
    def __init__(self, s_dim, a_dim, hidden_size=64):
        super(PPOActor, self).__init__()
        self.actions_mean = nn.Sequential(
            nn.Linear(s_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, a_dim)
        )
        # Learnable parameter for log standard deviation of actions
        self.actions_logstd = nn.Parameter(torch.zeros(a_dim))

    def forward(self, state, deterministic=False):
        actions_mean = self.actions_mean(state)
        actions_std = torch.exp(self.actions_logstd)
        dist = Dist(actions_mean, actions_std)
        
        if deterministic:
            action = actions_mean
        else:
            action = dist.sample()
            
        return action, dist.log_prob(action).sum(-1)

    def evaluate(self, state, action):
        actions_mean = self.actions_mean(state)
        actions_std = torch.exp(self.actions_logstd)
        dist = Dist(actions_mean, actions_std)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)


class PPOCritic(nn.Module):
    """Critic network to estimate the state value function. A simple MLP.
    
    Args:
        s_dim: State dimension.
        hidden_size: Number of hidden units in each layer.
    """
    def __init__(self, s_dim, hidden_size=64):
        super(PPOCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.model(state)[:, 0]


class PPO(base.Agent):
    """PPO class

    Warning:
        You can use this if you want but honestly just use the StableBaselines3 implementation.

    Args:
        actor: Actor network.
        critic: Critic network.
        lr_policy: Learning rate for the policy network.
        lr_critic: Learning rate for the critic network.
        max_grad_norm: Maximum gradient norm for clipping.
        ent_weight: Entropy weight for exploration.
        clip_val: Clipping value for PPO.
        sample_n_epoch: Number of epochs to sample minibatches.
        sample_mb_size: Size of each minibatch.
        device: Device to run the computations on.
    """
    def __init__(self, actor, critic, lr_policy=3e-4, lr_critic=4e-4, max_grad_norm=0.5, 
                ent_weight=0.0, clip_val=0.2, sample_n_epoch=10, sample_mb_size=32, device='cpu'):
        

        ################# OPTIMIZERS #################
        self.opt_policy = torch.optim.Adam(actor.parameters(), lr_policy, eps=1e-5) # was 1-e5
        self.opt_value = torch.optim.Adam(critic.parameters(), lr_critic, eps=1e-5)


        ############## MODELS################
        self.actor = actor
        self.critic = critic

        self.actor.to(device)
        self.critic.to(device)



        ############################# HYPERPARAMETERS #############################
        self.max_grad_norm = max_grad_norm  # Maximum gradient norm for clipping
        self.ent_weight = ent_weight  # Entropy weight for exploration
        self.clip_val = clip_val  # Clipping value for PPO
        self.sample_n_epoch = sample_n_epoch  # Number of epochs to sample minibatches
        self.sample_mb_size = sample_mb_size  # Size of each minibatch
        self.device = device
        
        


    def train(self, states, actions, prev_val, advantages, returns, prev_lobprobs):
        """Train the PPO model using provided experience. 

        Args:
            states: State samples.
            actions: Action samples.
            prev_val: Previous state value estimates.
            advantages: Advantage estimates.
            returns: Discounted return estimates.
            prev_lobprobs: Previous log probabilities of actions.
        Returns:
            pg_loss: Policy loss.
            v_loss: Value loss.
            entropy: Average entropy.
        """

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)


        advantages = torch.from_numpy(advantages).to(self.device)
        returns = torch.from_numpy(returns).to(self.device)

        prev_val = torch.from_numpy(prev_val).to(self.device)
       
        prev_lobprobs = torch.from_numpy(prev_lobprobs).to(self.device)
        
        episode_length = len(states)
        indices = np.arange(episode_length)
        

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.sample_n_epoch):
            np.random.shuffle(indices)
            
            for start_idx in range(0, episode_length, self.sample_mb_size):

                # Get minibatch indices
                end_idx = min(start_idx + self.sample_mb_size, episode_length)
                minibatch_indices = indices[start_idx:end_idx]
                
                # Sample minibatch
                sample_states = states[minibatch_indices]
                sample_actions = actions[minibatch_indices]
                sample_old_values = prev_val[minibatch_indices]
                sample_advs = advantages[minibatch_indices]
                sample_returns = returns[minibatch_indices]
                sample_old_a_logps = prev_lobprobs[minibatch_indices]

                # Policy loss
                sample_a_logps, entropy = self.actor.evaluate(sample_states, sample_actions)
                ratio = torch.exp(sample_a_logps - sample_old_a_logps)
                

                # Compute value loss with clipping
                pg_loss1 = -sample_advs * ratio
                pg_loss2 = -sample_advs * torch.clamp(ratio, 1.0 - self.clip_val, 1.0 + self.clip_val)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() - self.ent_weight * entropy.mean()

                # Value loss
                sample_values = self.critic(sample_states)
                v_pred_clip = sample_old_values + torch.clamp(
                    sample_values - sample_old_values, 
                    -self.clip_val, 
                    self.clip_val
                )
                v_loss1 = (sample_returns - sample_values).pow(2)
                v_loss2 = (sample_returns - v_pred_clip).pow(2)
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                # Update networks
                self.opt_policy.zero_grad()
                pg_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                self.opt_value.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_value.step()

        return pg_loss.item(), v_loss.item(), entropy.mean().item()
    
    def act(self, obs, *args, **kwargs):

        obs,_ = ns_gym.utils.type_mismatch_checker(obs,None)
        obs = ns_gym.nn_model_input_checker(obs)

        self.actor.eval()

        with torch.no_grad():
            action, _ = self.actor(obs,deterministic=True)

        return action
    
    def train_ppo(self, env,config):
        """Main training loop PPO algorithm.

        Saves best model based on running average reward over 100 episodes.
        
        Args: 
            env: Gym environment.
            config: Configuration dictionary.
        
        Returns:
            best_reward: Best running average reward over 100 episodes. 
        """
        # Initialize environment
        # s_dim = env.observation_space.shape[0]  # For now I am manully setting the state and action dimensions in the config file. 
        # a_dim = env.action_space.shape[0] # This method does not work all the time -- due to the dfiferent types of action/observation spaces in gym.

        if "s_dim" not in config:
            s_dim = env.observation_space.shape[0]
        else:
            s_dim = config["s_dim"]


        if "a_dim" not in config:
            a_dim = config["a_dim"]
        else:
            a_dim = env.action_space.shape[0]

        

        
        # Training parameters from Config
        max_episodes = config["max_episodes"]
        batch_size = config["batch_size"]
        minibatch_size = config["minibatch_size"]
        n_epochs = config["n_epochs"]
        hidden_size = config["hidden_size"] # was 64
        max_steps = config["max_steps"]
        gamma = config["gamma"]
        lamb = config["lamb"]
        device = config["device"]

        # Train 

        lr_policy = config["lr_policy"]
        lr_critic = config["lr_critic"]
        max_grad_norm = config["max_grad_norm"]
        clip_val = config["clip_val"]
        ent_weight = config["ent_weight"]


        # Model save path
        save_path = config["save_path"]
        

        best_reward = -np.inf
        # Initialize PPO agent
        # actor = Actor(s_dim=s_dim, a_dim=a_dim, hidden_size=hidden_size)
        # critic = Critic(s_dim=s_dim, hidden_size=hidden_size)


        
        # agent = PPO(actor, critic, lr_policy=lr_policy, lr_critic=lr_critic, max_grad_norm=max_grad_norm,clip_val=clip_val, ent_weight=ent_weight, sample_n_epoch=n_epochs, sample_mb_size=minibatch_size, device=device)
        
        buffers =  initialize_buffers(s_dim, a_dim, max_steps)
        #runner = EnvRunner(s_dim, a_dim, gamma=0.99, lamb=0.8, max_step=batch_size)

        # Metrics storage
        rewards_history = []
        losses_history = []
        running_rewards = deque(maxlen=100)
        start_time = time.time()

        for i in range(max_episodes):
            # Run episode to collect data using GAE
            with torch.no_grad():
                mb_states, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = \
                    run_environment(env,buffers,self.actor, self.critic, s_dim, a_dim, max_steps,gamma, lamb, device)
                
                # Use GAE-Lambda advantage estimation
                last_value = self.critic(
                    torch.tensor(np.expand_dims(mb_states[-1], axis=0), dtype=torch.float32).to(self.device)
                ).detach().cpu().numpy()
                
                mb_returns = compute_gae(mb_rewards, mb_values,gamma,lamb,last_value)
                mb_advs = mb_returns - mb_values

            # Train using minibatches
            pg_loss, v_loss, ent = self.train(
                mb_states, mb_actions, mb_values, mb_advs, mb_returns, mb_old_a_logps
            )
            
            # Store metrics
            episode_reward = mb_rewards.sum()
            rewards_history.append(episode_reward)
            losses_history.append(pg_loss + v_loss)
            running_rewards.append(episode_reward)

            mean_reward = np.mean(running_rewards) if len(running_rewards) == 100 else np.mean(rewards_history)
            print(f"[Episode {i:4d}] reward = {episode_reward:.1f}, mean_100 = {mean_reward:.1f}, "
                f"pg_loss = {pg_loss:.3f}, v_loss = {v_loss:.3f}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(self.actor.state_dict(), save_path + f'{config["env_name"]}_actor_weights.pt')
                torch.save(self.critic.state_dict(),  save_path + f'{config["env_name"]}_critic_weights.pt')

            # # Check if solved
            # if len(running_rewards) == 100 and mean_reward >= 300:
            #     print("\nEnvironment solved! Saving final model...")
            #     torch.save(actor.state_dict(), 'bipedalwalker_actor_weights_keplinns.pt')
            #     torch.save(critic.state_dict(), 'bipedalwalker_critic_weights_keplinns.pt')
            #     break
        
        return best_reward


