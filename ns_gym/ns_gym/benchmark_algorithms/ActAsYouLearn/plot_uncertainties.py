import pickle

with open('results/{}_network_weights_itr_3'.format("discretegrid"), 'rb') as f1:
    network_weights1 = pickle.load(f1)

with open('results/{}_latent_weights_itr_3'.format("discretegrid"), 'rb') as f3:
    latent_weights1 = pickle.load(f3)

weight_set1 = latent_weights1.reshape(latent_weights1.shape[1],)

