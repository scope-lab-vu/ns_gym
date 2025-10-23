import gymnasium as gym
import ns_gym as nsb 


def make_continous_env(gym_config,wrapper_config):
    """Make non-stationary Classic Control environment.
    """


    env = gym.make("CliffWalking-v0",max_episode_steps=gym_config["max_episode_steps"])

    change_notification = wrapper_config["change_notification"]
    delta_change_notification = wrapper_config["delta_change_notification"]
    in_sim_change = wrapper_config["in_sim_change"]
    param_name = wrapper_config["param_name"] 
    terminal_cliff = wrapper_config["terminal_cliff"]
    modified_rewards = None


    scheduler = nsb.schedulers.ContinuousScheduler(start=0,end=10)
    updateFn = nsb.update_functions.DistributionDecrmentUpdate(scheduler,k=0.02)
    params = {param_name:updateFn}
    realworld_env = nsb.wrappers.NSCliffWalkingWrapper(env,
                                                         params,
                                                         change_notification=change_notification,
                                                         delta_change_notification=delta_change_notification,
                                                         in_sim_change=in_sim_change,
                                                         terminal_cliff=terminal_cliff,
                                                         modified_rewards=modified_rewards)
    return realworld_env


def make_discrete_env(p,gym_config,wrapper_config):
    """Make non-stationary Classic Control environment.
    """

    env = gym.make("CliffWalking-v0",max_episode_steps=gym_config["max_episode_steps"])

    change_notification = wrapper_config["change_notification"]
    delta_change_notification = wrapper_config["delta_change_notification"]
    in_sim_change = wrapper_config["in_sim_change"]
    param_name = wrapper_config["param_name"] 
    terminal_cliff = wrapper_config["terminal_cliff"]
    modified_rewards = None

    scheduler = nsb.schedulers.DiscreteScheduler({0})
    updateFn = nsb.update_functions.DistributionStepWiseUpdate(scheduler,update_values=[[p,(1-p)/3,(1-p)/3,(1-p)/3]])
    params = {param_name:updateFn}
    realworld_env = nsb.wrappers.NSCliffWalkingWrapper(env,
                                                         params,
                                                         change_notification=change_notification,
                                                         delta_change_notification=delta_change_notification,
                                                         in_sim_change=in_sim_change,
                                                         terminal_cliff=terminal_cliff,
                                                         modified_rewards=modified_rewards)
    return realworld_env


