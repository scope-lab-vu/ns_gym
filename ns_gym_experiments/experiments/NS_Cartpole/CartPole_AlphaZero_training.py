import yaml
import os
import gymnasium as gym
import ns_gym as nsb
from ns_gym.benchmark_algorithms.AlphaZero.alphazero import AlphaZeroAgent, AlphaZeroNetwork

def train_alpha_zero(config_file_path):
    """Train the AlphaZero agent using yaml configuration file

    Args:
        config_file_path (_type_): _description_
    """

    with open(config_file_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    ########## SET UP TRAINING ENVIRONMENT ##########

    env_config = config["gym_env"]
    wrapper_config = config["wrapper"]
    agent_config = config["agent"]
    training_config = config["training"]

    #################################################

    ########### make the environment ##############

    env = gym.make(env_config["name"],
                   max_episode_steps=env_config["max_episode_steps"])
    
    scheduler = nsb.schedulers.ContinuousScheduler()
    updateFn = nsb.update_functions.NoUpdate(scheduler=scheduler)
    param = wrapper_config["param"]

    env = nsb.wrappers.NSClassicControlWrapper(env,
                                               {param:updateFn},
                                               change_notification=wrapper_config["change_notification"],
                                               delta_change_notification=wrapper_config["delta_change_notification"],
                                               in_sim_change=wrapper_config["in_sim_change"])
    

    #################################################

    ########### make the agent #####################
    

    alphazero_agent = AlphaZeroAgent(action_space_dim=agent_config["action_space_dim"],
                                     observation_space_dim=agent_config["observation_space_dim"],
                                     model=AlphaZeroNetwork,
                                     n_hidden_layers=agent_config["n_hidden_layers"],
                                     n_hidden_units=agent_config["n_hidden_units"],
                                     gamma=agent_config["gamma"],
                                     c=agent_config["c"],
                                     num_mcts_simulations=agent_config["num_mcts_simulations"],
                                     max_mcts_search_depth=agent_config["max_mcts_search_depth"],
                                     )
    

    #################################################

    ########### TRAIN THE AGENT ####################

    episode_rewards = alphazero_agent.train(env,
                          training_config["num_episodes"],
                          lr=training_config["lr"],
                          max_episode_len=training_config["max_episode_len"],
                          batch_size=training_config["batch_size"],
                          n_epochs=training_config["n_epochs"],
                          experiment_name=training_config["experiment_name"],
                          weight_decay=training_config["weight_decay"])
    



if __name__ == "__main__":
    config_file_path = "/media/--/home/n--/ns_gym/experiments/NS_Cartpole/configs/CartPole_AlphaZero_training_config.yaml"
    train_alpha_zero(config_file_path)

