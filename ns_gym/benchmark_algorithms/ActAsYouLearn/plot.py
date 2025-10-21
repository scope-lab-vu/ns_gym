import pickle
import numpy as np
import matplotlib.pyplot as plt
domain = 'discretegrid'
#%%
#with open('results/record/{}_results_latent1'.format(domain),'rb') as f:
#    results = pickle.load(f)
with open('results/{}_results'.format(domain),'rb') as f:
    results = pickle.load(f)
#%% md
## Plot Results
#%%
# Group rewards, errors by instance
reward_key = 'Reward'
error_key = 'BNN Error'
clean_results = {(0,reward_key):[],(0,error_key):[]}
for test_inst in [0]:
    for run in range(2):
        clean_results[(test_inst,reward_key)].append(results[(test_inst,run)][3])
        clean_results[(test_inst,error_key)].append(results[(test_inst,run)][6])
    clean_results[(test_inst,reward_key)] = np.vstack(clean_results[(test_inst,reward_key)])
    clean_results[(test_inst,error_key)] = np.vstack(clean_results[(test_inst,error_key)])

#%%
def plot_results(clean_results, test_inst):
    f, ax_array = plt.subplots(2,figsize=(7,7))
    result_names = [reward_key,error_key]
    for result_idx in range(2):
        result = result_names[result_idx]
        mean_result = np.mean(clean_results[(test_inst,result)], axis=0)
        std_result = np.std(clean_results[(test_inst,result)], axis=0)
        ax_array[result_idx].errorbar(x=np.arange(len(mean_result)),y=mean_result,yerr=std_result)
        _ = ax_array[result_idx].set_ylim((np.min(mean_result)-0.01,np.max(mean_result)+0.01))
        ax_array[result_idx].set_ylabel(result)
    ax_array[1].set_xlabel("Episode")
    f.suptitle("Full HiP-MDP Training Results Instance {}".format(test_inst),fontsize=12)
    plt.show()  # This line will show the plots in a standalone window.

#%%
plot_results(clean_results, 0)
#%%
#plot_results(clean_results, 1)