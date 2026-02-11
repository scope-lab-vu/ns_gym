import ns_gym
import ns_gym.base as base
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time

def _observation_to_vector(observation):
    """Convert an observation to a flat float32 vector."""
    observation, _ = ns_gym.utils.type_mismatch_checker(observation, None)
    if isinstance(observation, torch.Tensor):
        obs_arr = observation.detach().cpu().numpy()
    else:
        obs_arr = np.asarray(observation)
    obs_arr = obs_arr.astype(np.float32, copy=False)
    if obs_arr.ndim == 0:
        obs_arr = np.expand_dims(obs_arr, axis=0)
    return obs_arr.reshape(-1)

def initialize_buffers(state_dim, action_dim, max_steps, is_discrete=False):
    """Initialize buffers for states, actions, values, rewards, and log probabilities."""
    actions_shape = (max_steps,) if is_discrete else (max_steps, action_dim)
    actions_dtype = np.int64 if is_discrete else np.float32
    buffers = {
        "states": np.zeros((max_steps, state_dim), dtype=np.float32),
        "actions": np.zeros(actions_shape, dtype=actions_dtype),
        "values": np.zeros((max_steps,), dtype=np.float32),
        "rewards": np.zeros((max_steps,), dtype=np.float32),
        "log_probs": np.zeros((max_steps,), dtype=np.float32),
    }
    return buffers

def compute_discounted_returns(rewards, gamma, last_value):
    """Compute discounted returns."""
    last_value = float(np.asarray(last_value).squeeze())
    returns = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            returns[t] = rewards[t] + gamma * last_value
        else:
            returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns

def compute_gae(rewards, values, gamma, lamb, last_value):
    """Compute Generalized Advantage Estimation (GAE)."""
    last_value = float(np.asarray(last_value).squeeze())
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else float(values[t + 1])
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lamb * last_gae
    return advantages + values

def run_environment(
    env,
    buffers,
    policy_net,
    value_net,
    state_dim,
    action_dim,
    max_steps,
    gamma,
    lamb,
    device,
    is_discrete_action=False,
):
    """
    Run an episode in the environment, collecting states, actions, rewards, and other data.
    """
    state = _observation_to_vector(env.reset()[0])
    episode_length = max_steps

    for step in range(max_steps):
        state_tensor = torch.tensor(state[None, :], dtype=torch.float32, device=device)
        action, log_prob = policy_net(state_tensor)
        value = value_net(state_tensor)

        # Store data in buffers
        buffers["states"][step] = state
        # For discrete action spaces, convert the action tensor to an int index.
        # This undoes the _observation_to_vector.
        if is_discrete_action:
            env_action = int(action.detach().cpu().item())
            buffers["actions"][step] = env_action
        else:
            env_action = action.detach().cpu().numpy()[0]
            buffers["actions"][step] = env_action
        buffers["log_probs"][step] = float(log_prob.detach().cpu().item())
        buffers["values"][step] = float(value.detach().cpu().item())

        # Take a step in the environment
        state, reward, terminated, truncated, _ = env.step(env_action)
        state, reward = ns_gym.utils.type_mismatch_checker(state, reward)
        state = _observation_to_vector(state)
        buffers["rewards"][step] = float(reward)
        if terminated or truncated:
            episode_length = step + 1
            break

    # Compute the returns
    last_state_tensor = torch.tensor(state[None, :], dtype=torch.float32, device=device)
    last_value = float(value_net(last_state_tensor).detach().cpu().item())
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

    Supports both continuous and discrete action spaces.

    Args:
        s_dim: State dimension.
        a_dim: Action dimension.
        hidden_size: Number of hidden units in each layer.
        is_discrete: Whether the action space is discrete.
    """
    def __init__(self, s_dim, a_dim, hidden_size=64, is_discrete=False):
        super(PPOActor, self).__init__()
        self.is_discrete = is_discrete
        self.policy_model = nn.Sequential(
            nn.Linear(s_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, a_dim),
        )
        if not self.is_discrete:
            # Learnable parameter for log standard deviation of actions.
            self.actions_logstd = nn.Parameter(torch.zeros(a_dim))

    def forward(self, state, deterministic=False):
        policy_output = self.policy_model(state)

        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=policy_output)
            if deterministic:
                action = torch.argmax(policy_output, dim=-1)
            else:
                action = dist.sample()
            return action, dist.log_prob(action)

        actions_std = torch.exp(self.actions_logstd)
        dist = Dist(policy_output, actions_std)

        if deterministic:
            action = policy_output
        else:
            action = dist.sample()

        return action, dist.log_prob(action).sum(-1)

    def evaluate(self, state, action):
        policy_output = self.policy_model(state)
        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=policy_output)
            action = action.long().view(-1)
            return dist.log_prob(action), dist.entropy()

        actions_std = torch.exp(self.actions_logstd)
        dist = Dist(policy_output, actions_std)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
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
        self.is_discrete = getattr(actor, "is_discrete", False)

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

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        if self.is_discrete:
            actions = actions.long()
        else:
            actions = actions.float()


        advantages = torch.from_numpy(advantages).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)

        prev_val = torch.from_numpy(prev_val).float().to(self.device)
       
        prev_lobprobs = torch.from_numpy(prev_lobprobs).float().to(self.device)
        
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
        obs = _observation_to_vector(obs)
        obs = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)

        self.actor.eval()

        with torch.no_grad():
            action, _ = self.actor(obs,deterministic=True)

        return action.squeeze(0).cpu()
    
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

        # Both problems above resolved :)

        if "s_dim" not in config:
            if hasattr(env.observation_space, "shape") and env.observation_space.shape is not None and len(env.observation_space.shape) > 0:
                s_dim = int(np.prod(env.observation_space.shape))
            else:
                s_dim = 1
        else:
            s_dim = config["s_dim"]


        is_discrete_action = hasattr(env.action_space, "n")
        if "a_dim" not in config:
            if is_discrete_action:
                a_dim = int(env.action_space.n)
            elif hasattr(env.action_space, "shape") and env.action_space.shape is not None and len(env.action_space.shape) > 0:
                a_dim = int(np.prod(env.action_space.shape))
            else:
                raise ValueError("Unsupported action space. PPO currently supports Discrete and Box spaces.")
        else:
            a_dim = config["a_dim"]

        if self.is_discrete != is_discrete_action:
            raise ValueError(
                "Actor action type does not match environment action space. "
                "Initialize PPOActor with is_discrete=True for Discrete action spaces."
            )
        

        
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
        
        buffers = initialize_buffers(s_dim, a_dim, max_steps, is_discrete=is_discrete_action)
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
                    run_environment(
                        env,
                        buffers,
                        self.actor,
                        self.critic,
                        s_dim,
                        a_dim,
                        max_steps,
                        gamma,
                        lamb,
                        device,
                        is_discrete_action=is_discrete_action,
                    )
                
                # Use GAE-Lambda advantage estimation
                last_value = self.critic(
                    torch.tensor(np.expand_dims(mb_states[-1], axis=0), dtype=torch.float32).to(self.device)
                ).detach().cpu().item()
                
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

