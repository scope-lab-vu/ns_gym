import gymnasium as gym
import ns_gym.schedulers as schedulers
import ns_gym.update_functions as update_functions
from ns_gym.benchmark_algorithms.AlphaZero.alphazero import AlphaZeroAgent,AlphaZeroNetwork
import ns_gym as nsb

env = gym.make("CliffWalking-v0",render_mode="ansi")
scheduler = schedulers.ContinuousScheduler(start=0,end=10)
#update_fn = update_functions.LCBoundedDistrubutionUpdate(scheduler,L=1) ### Test a sequence of L values.gitp
update_fn = update_functions.DistributionDecrmentUpdate(scheduler,k=0.02)
# update_fn = update_functions.DistributionNoUpdate(scheduler)

dist = [1,0,0,0]
print(dist)

env = nsb.wrappers.NSCliffWalkingWrapper(env, tunable_params={"P": update_fn}, change_notification=True, delta_change_notification=False, in_sim_change=False)
model_path = "/home/cc/ns_gym/experiments/NS_CliffWalking/Deterministic_Cliff_Walking_3_layers_96_units_trial_1_best_model_checkpoint.pth"
action_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

model_path = ""
total_reward = []
for i in range(10):
    obs,_ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    print(f"############### EPISODE {i} ###############")
    print(env.render())

    
    alphazero_agent = AlphaZeroAgent(action_space_dim=4,
                                        observation_space_dim=48,
                                        model=AlphaZeroNetwork,
                                        lr=0.001,
                                        n_hidden_layers=3,
                                        n_hidden_units=96,
                                        gamma=0.99,
                                        c=np.sqrt(2),
                                        num_mcts_simulations=1000,
                                        max_mcts_search_depth=50,
                                        model_checkpoint_path="--",
                                        )

    while not done and not truncated:
        planning_env = env.get_planning_env()
        # mcts_agent = nsb.benchmark_algorithms.MCTS(env, obs,d=20, m=150, c=1, gamma=0.8)
        # action,action_values = mcts_agent.search()
        # print(action_values)
        action = alphazero_agent.act(obs,env)
        
        print(f"\rAction: {action_map[action]}") 

        # action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(env.transition_prob)
        # print(5*"-")
        print("\r"+env.render())
        episode_reward += reward.reward
    print(f"Episode reward: {episode_reward}")
    total_reward.append(episode_reward)




