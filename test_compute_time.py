from ns_gym.wrappers import NSFrozenLakeWrapper,NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler,DiscreteScheduler
from ns_gym.update_functions import DistributionDecrementUpdate,DecrementUpdate

import gymnasium as gym

import time
import numpy as np
from ns_gym.benchmark_algorithms import MCTS
from ns_gym.utils import type_mismatch_checker




num_steps = 1000
env = gym.make('FrozenLake-v1',render_mode="rgb_array",is_slippery=False)
#env = gym.make("CartPole-v1")

### Define the scheduler #######
scheduler = ContinuousScheduler(start=1,end=1) #Update the slipperiness at each timestep

#### Define the update function #####
update_function = DistributionDecrementUpdate(scheduler=scheduler,k = 0.00001) #Decrement the slipperiness by 0.1 at each timestep where the scheduler fires true
#update_function = DecrementUpdate(scheduler=scheduler,k=0.1)
# Map parameter to update function
####### Defining environmental parameters ############
param = "P"
#param = "masspole"
params = {param:update_function}
###### Import the frozen lake
env = NSFrozenLakeWrapper(env,params, initial_prob_dist=[1,0,0])
#env = NSClassicControlWrapper(env,params,change_notification=True,delta_change_notification=True)

obs,_ = env.reset()
type_mismatch_checker(obs,reward=None)

agent = MCTS(env,obs,25,100,1.44,0.9)

actions = [0,1]

print("start")

time_list = [ ]
for i in range(num_steps):
    start = time.time()
    planning_env = env.get_planning_env()
    end = time.time()
    time_list.append(end-start)
    # action = agent.act(obs,env=env)
    action = np.random.choice(actions)
    
    
    obs,reward,done,truncated,info = env.step(action)
    # obs,reward = type_mismatch_checker(obs,reward)

    if done or truncated:   
        obs,_ = env.reset()
        # type_mismatch_checker(obs,reward=None)


print(f"copy time: ", np.mean(time_list))
#print(f"Average step time {(end-start)/num_steps}")

# Ns-gym is 409.3% slower doing copies for MCTS...

# ns_gym Average step time 0.11881487703323364 ( no plannging env)
# ns_gym Average step time 0.11929670596122742 (with planing env)
# gym average step time 0.029031862020492554



# Doing random actions gym Average step time 8.472919464111327e-06
# Doing random actions ns_gym Average step time 0.00014066195487976075
# with get planning env averarge step time 0.0012685842514038087

#Cartpole 
# copy time:  0.00023029565811157225


        
#copy time:  0.0011184265613555908
    








