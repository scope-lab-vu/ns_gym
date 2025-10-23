import gymnasium as gym
import ns_gym 
from ns_gym.benchmark_algorithms.AlphaZero.alphazero import AlphaZeroAgent, AlphaZeroNetwork




############## Env Setup ################
gym = gym.make("ns_gym/Bridge-v0",max_episode_steps=100)
scheduler = ns_gym.schedulers.ContinuousScheduler()
updateFn = ns_gym.update_functions.DistributionDecrmentUpdate(scheduler,k=0.02)
tunable_params = {"P":updateFn}
env = ns_gym.wrappers.NSBridgeWrapper(gym,
                                        tunable_params=tunable_params,
                                        change_notification=True,
                                        delta_change_notification=True,
                                        initial_prob_dist=[1,0,0])




######## Agent/Env Interaction ##########
done = False
truncated = False

obs,_ = env.reset()

while not done and not truncated:
    planner_env = env.get_planning_env()
    agent = ns_gym.benchmark_algorithms.MCTS(env=planner_env,state=obs,d=100,m=100,c=1.44,gamma=0.99)
    action, _ = agent.search()
    obs,reward,done,truncated,info = env.step(action)
    env.render()


############ To get the transition matrix ############
print("Transition Matrix")
print()
print(env.transition_matrix)

