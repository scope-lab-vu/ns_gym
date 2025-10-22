import numpy as np
import rats
import gymnasium as gym
from ns_gym import wrappers, schedulers, update_functions

def add_action(actions, action):
    if action == 0:
        actions.append('Left')
    elif action == 1:
        actions.append('Down')
    elif action == 2:
        actions.append('Right')
    elif action == 3:
        actions.append('Up')
    else:
        print('Error')
        exit()
    return actions


def main():
    np.random.seed(1993)
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    # env = gym.make("CartPole-v1")
    fl_scheduler = schedulers.ContinuousScheduler()
    fl_update_fn = update_functions.DistributionNoUpdate(fl_scheduler)
    # cartpole_update_fn = update_functions.NoUpdate(fl_scheduler)
    param = {"P": fl_update_fn}
    # param = {"gravity":cartpole_update_fn}
    env2 = wrappers.NSFrozenLakeWrapper(env, param, change_notification=True, delta_change_notification=True,
                                       in_sim_change=False, initial_prob_dist=[1, 0, 0])
    #print("check P",env.transition_prob)
    # Parameters
    # Define the number of simulations
    #num_simulations = 10000
    # Track the outcomes in a dictionary
    #outcome_counts = {0: 0, 1: 0, 4: 0}
    # for _ in range(num_simulations):
    #     # Set the environment to the fixed state
    #     env.reset()  # Reset the environment first if needed
    #
    #     # Take a fixed step in the environment
    #     next_state, reward, done, _, info = env.step(1)
    #     # Update outcome counts based on the observed new state
    #     outcome_counts[next_state.state] += 1
    # print(outcome_counts)
    #env = nsg.NSBridgeV0()
    #env = nsf.NSFrozenLakeV0()
    #env.set_epsilon(0.5)
    #env.step(action)
    depth = 3
    agent = rats.RATS(env.action_space, gamma=0.95, max_depth=depth)
    #agent = asyndp.AsynDP(env.action_space, gamma=0.9, max_depth=depth, is_model_dynamic=False)

    agent.display()
    total_reward = 0
    total_penalty = 0
    # Run
    for i in range(10):
        render = True
        actions = []
        done = False
        env.reset()
        if render:
            print(env.render())
        timeout = 1000
        undiscounted_return, total_time, discounted_return = 0.0, 0, 0.0
        #env.reset(1, i)
        for t in range(timeout):
            action = agent.act(env2, done)
            actions = add_action(actions, action)
            #_, reward, done, _ = env.step(action)
            next_state, reward, done, _, info = env.step(action)
            #reward = reward.reward
            undiscounted_return += reward
            discounted_return += (agent.gamma ** t) * reward
            if render:
                print(env.render())
            if (t + 1 == timeout) or done:
                total_time = t + 1
                break
        if reward == 1:
            total_reward += 1
        if reward == -1:
            total_penalty += -1
        print('End of episode')
        print('Total time          :', total_time)
        print('Actions             :', actions)
        print('Discounted return   :', discounted_return)
        print('Un-discounted return :', undiscounted_return)
        print('total reward so far:', total_reward)
        print('total penalty so far:', total_penalty)

if __name__ == "__main__":
    main()
