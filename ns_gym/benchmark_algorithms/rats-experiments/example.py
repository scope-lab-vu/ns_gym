import numpy as np
import gymnasium as gym

import rats
from ns_gym import schedulers, update_functions, wrappers

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
    fl_scheduler = schedulers.PeriodicScheduler(period=3)
    fl_update_fn = update_functions.DistributionDecrementUpdate(
        scheduler=fl_scheduler,
        k=0.01,
    )
    param = {"P": fl_update_fn}
    env = wrappers.NSFrozenLakeWrapper(
        env,
        param,
        change_notification=True,
        delta_change_notification=True,
        in_sim_change=False,
        initial_prob_dist=[1, 0, 0],
    )

    depth = 3
    agent = rats.RATS(env.action_space, gamma=0.95, max_depth=depth)

    agent.display()
    total_reward = 0
    total_penalty = 0
    render = False
    for episode in range(10):
        actions = []
        done = False
        truncated = False
        _, _ = env.reset(seed=episode)
        if render:
            print(env.render())
        timeout = 1000
        undiscounted_return, total_time, discounted_return = 0.0, 0, 0.0
        for t in range(timeout):
            action = agent.act(env, done)
            actions = add_action(actions, action)
            _, reward, done, truncated, info = env.step(action)
            reward_value = reward.reward if hasattr(reward, "reward") else reward
            undiscounted_return += reward_value
            discounted_return += (agent.gamma ** t) * reward_value
            if render:
                print(env.render())
            if (t + 1 == timeout) or done or truncated:
                total_time = t + 1
                break
        if reward_value == 1:
            total_reward += 1
        if reward_value == -1:
            total_penalty += -1
        print(f'End of episode {episode}')
        print('Total time          :', total_time)
        print('Actions             :', actions)
        print('Discounted return   :', discounted_return)
        print('Un-discounted return :', undiscounted_return)
        print('Transition probs    :', info.get("transition_prob"))
        print('total reward so far:', total_reward)
        print('total penalty so far:', total_penalty)

if __name__ == "__main__":
    main()
