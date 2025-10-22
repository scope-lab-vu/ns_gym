import pickle
import os
import autograd.numpy as np
import seaborn as sns
sns.set_style("whitegrid",{'axes.grid' : False})

from BayesianNeuralNetwork import *
from HiPMDP import HiPMDP
#from __future__ import print_function

if not os.path.isdir('./results'):
    os.mkdir('results')

#domain = 'discretegrid'
domain = 'frozenlake'
run_type = 'modelfree'
num_batch_instances = 1
#preset_hidden_params = [{'latent_code': 1}, {'latent_code': 2}]
preset_hidden_params = [{'latent_code': 2}]
ddqn_learning_rate = 0.0001
episode_count = 10000
bnn_hidden_layer_size = 25
bnn_num_hidden_layers = 3
bnn_network_weights = None
eps_min = 0.1
test_inst = None
create_exp_batch = True
state_diffs = False
grid_beta = 0.1
batch_generator_hipmdp = HiPMDP(domain, preset_hidden_params,
                                ddqn_learning_rate=ddqn_learning_rate,
                                episode_count=episode_count,
                                run_type=run_type, eps_min=eps_min,
                                create_exp_batch=create_exp_batch,
                                num_batch_instances=num_batch_instances,
                                grid_beta=grid_beta,
                                print_output=True)

## Example code of how to initialize a batch generator for HIV or Acrobot
# domain = 'hiv' # 'acrobot'
# with open('preset_parameters/'+domain+'_preset_hidden_params','r') as f:
#     preset_parameters = pickle.load(f)

# run_type = 'modelfree'
# num_batch_instances = 5
# preset_hidden_params = preset_parameters[:num_batch_instances]

#(exp_buffer, networkweights, rewards, avg_rwd_per_ep, full_task_weights,
#     sys_param_set, mean_episode_errors, std_episode_errors) = batch_generator_hipmdp.run_experiment()
#%%
#with open('results/{}_exp_buffer'.format(domain),'wb') as f:
#    pickle.dump(exp_buffer,f)
#%%


with open('results/{}_exp_buffer'.format(domain),'rb') as f:
    exp_buffer = pickle.load(f)
#with open('data_buffer/frozenlake_2_exp_buffer','rb') as f:
#    exp_buffer = pickle.load(f)
#print(exp_buffer)
# Create numpy array
exp_buffer_np = np.vstack(exp_buffer)
# Collect the instances that each transition came from
inst_indices = exp_buffer_np[:,4]
inst_indices = inst_indices.astype(int)
# Group experiences by instance
# Create dictionary where keys are instance indexes and values are np.arrays experiences
exp_dict = {}
for idx in range(batch_generator_hipmdp.instance_count):
    exp_dict[idx] = exp_buffer_np[inst_indices == idx]
X = np.array([np.hstack([exp_buffer_np[tt,0],exp_buffer_np[tt,1]]) for tt in range(exp_buffer_np.shape[0])])
y = np.array([exp_buffer_np[tt,3] for tt in range(exp_buffer_np.shape[0])])
num_dims = 2
num_actions = 4
num_wb = 5
if state_diffs:
    # subtract previous state
    y -= X[:,:num_dims]

relu = lambda x: np.maximum(x, 0.)
param_set = {
    'bnn_layer_sizes': [num_dims+num_actions+num_wb]+[bnn_hidden_layer_size]*bnn_num_hidden_layers+[num_dims*2],
    'weight_count': num_wb,
    'num_state_dims': num_dims,
    'bnn_num_samples': 50,
    'bnn_batch_size': 32,
    'num_strata_samples': 5,
    'bnn_training_epochs': 1,
    'bnn_v_prior': 1,
    'bnn_learning_rate': 0.0005,
    'bnn_alpha':0.5,
    'wb_num_epochs':1,
    'wb_learning_rate':0.0005
}
# Initialize latent weights for each instance
full_task_weights = np.random.normal(0.,0.1,(batch_generator_hipmdp.instance_count,num_wb))
# Initialize BNN
network = BayesianNeuralNetwork(param_set, nonlinearity=relu)
# Compute error before training
#with open('results/frozen_0.7/{}_network_weights_itr_3'.format(domain), 'rb') as f1:
#    network_weights1 = pickle.load(f1)
#    network_weights1 = pickle.load(f1)
#network.weights = network_weights1
l2_errors = network.get_td_error(np.hstack((X,full_task_weights[inst_indices])), y, location=0.0, scale=1.0, by_dim=False)
print ("Before training: Mean Error: {}, Std Error: {}".format(np.mean(l2_errors),np.std(l2_errors)))
np.mean(l2_errors),np.std(l2_errors)
#print ("L2 Difference in latent weights between instances: {}".format(np.sum((full_task_weights[0]-full_task_weights[1])**2)))
#exp_buffer_np = exp_buffer_np[np.random.choice(exp_buffer_np.shape[0], size=1000, replace=False)]
print(exp_buffer_np.shape)


def get_random_sample(start, stop, size):
    indices_set = set()
    while (len(indices_set) < size):
        indices_set.add(np.random.randint(start, stop))
        if len(indices_set) >= stop:
            break
    return np.array(list(indices_set))

checkpoint = 10
# size of sample to compute error on
sample_size = 1000
for i in range(20):
    # Update BNN network weights
    network.fit_network(exp_buffer_np, full_task_weights, 0, state_diffs=state_diffs,
                        use_all_exp=True)
    print('finished BNN update '+str(i))
    #print("checkpoint")
    if i % 1 == 0:
        #get random sample of indices
        sample_indices = get_random_sample(0,X.shape[0],sample_size)
        l2_errors = network.get_td_error(np.hstack((X[sample_indices],full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
        print ("After BNN update: iter: {}, Mean Error: {}, Std Error: {}".format(i,np.mean(l2_errors),np.std(l2_errors)))
    # Update latent weights
    #print("checkpoint1")
    for inst in np.random.permutation(batch_generator_hipmdp.instance_count):
        full_task_weights[inst,:] = network.optimize_latent_weighting_stochastic(
            exp_dict[inst],np.atleast_2d(full_task_weights[inst,:]),0,state_diffs=state_diffs,use_all_exp=True)
    #print("checkpoint2")
    print ('finished wb update '+str(i))
    #if np.mean(l2_errors) < checkpoint:
    #    with open('results/{}_network_weights'.format(domain), 'wb') as f:
    #        pickle.dump(network.weights, f)
    #    with open('results/{}_latent_weights'.format(domain), 'wb') as f:
    #        pickle.dump(full_task_weights, f)
    #    checkpoint = np.mean(l2_errors)
    # Compute error on sample of transitions
    if i % 1 == 0:
        #get random sample of indices
        sample_indices = get_random_sample(0,X.shape[0],sample_size)
        l2_errors = network.get_td_error(np.hstack((X[sample_indices],full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
        print ("After Latent update: iter: {}, Mean Error: {}, Std Error: {}".format(i,np.mean(l2_errors),np.std(l2_errors)))
        # We check to see if the latent updates are sufficiently different so as to avoid fitting [erroneously] to the same dynamics
        #print ("L2 Difference in latent weights between instances: {}".format(np.sum((full_task_weights[0]-full_task_weights[1])**2)))
    #if np.mean(l2_errors) < checkpoint:
    with open('results/{}_network_weights_itr_{}'.format(domain, i), 'wb') as f:
        pickle.dump(network.weights, f)
    with open('results/{}_latent_weights_itr_{}'.format(domain, i), 'wb') as f:
        pickle.dump(full_task_weights, f)
    #checkpoint = np.mean(l2_errors)
#%%
network_weights = network.weights
#%%
#with open('results/{}_network_weights'.format(domain), 'wb') as f:
#    pickle.dump(network.weights, f)
#with open('results/{}_latent_weights'.format(domain), 'wb') as f:
#    pickle.dump(full_task_weights, f)



