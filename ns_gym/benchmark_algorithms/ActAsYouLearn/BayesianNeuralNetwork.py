"""
Bayesian Neural Network Implementation for HiP-MDP.
Adapted from original code by Jose Miguel Hernandez Lobato,
following https://arxiv.org/abs/1605.07127
"""
from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
#from scipy.special import logsumexp
from autograd import grad
import math

class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

    def get_indexes(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return idxs

class BayesianNeuralNetwork(object):
    """Meta-class to handle much of the neccessary computations for BNN inference."""
    def __init__(self, param_set, nonlinearity, linear_latent_weights=False):
        """Initialize BNN.

        Arguments:
        param_set -- a dictionary containing the following keys:
            bnn_layer_sizes -- number of units in each layer including input and output
            weight_count -- numbber of latent weights
            num_state_dims -- state dimension
            bnn_num_samples -- number of samples of network weights drawn to approximate transitions
            bnn_alpha -- alpha divergence parameter
            bnn_batch_size -- minibatch size
            num_strata_samples -- number of transitions to draw from each strata in the prioritized replay
            bnn_training_epochs -- number of epochs of SGD in updating BNN network weights
            bnn_v_prior -- prior on the variance of the BNN weights
            bnn_learning_rate -- learning rate for BNN network weights
            wb_learning_rate -- learning rate for latent weights
            wb_num_epochs -- number of epochs of SGD in updating latent weights
        nonlinearity -- activation function for hidden layers

        Keyword arguments:
        linear_latent_weights -- boolean indicating whether to use linear top latent weights or embedded latent weights (default: False)
        """
        # Initialization is an adaptation of make_nn_funs() from
        # https://github.com/HIPS/autograd/blob/master/examples/bayesian_neural_net.py
        layer_sizes = param_set['bnn_layer_sizes']
        #print(layer_sizes)

        self.shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        #print(self.shapes)
        self.num_weights = sum((m+1) * n for m,n in self.shapes)
        self.linear_latent_weights = linear_latent_weights
        self.parser = WeightsParser()
        self.parser.add_shape('mean', (self.num_weights,1))
        self.parser.add_shape('log_variance', (self.num_weights,1))
        self.parser.add_shape('log_v_noise', (1,1))
        w = 0.1 * np.random.randn(self.parser.num_weights)
        #w[self.parser.get_indexes(w,'log_variance')] = -10.0
        #w[self.parser.get_indexes(w, 'log_v_noise')] = np.log(10.0)
        w[self.parser.get_indexes(w,'log_variance')] = -5
        w[self.parser.get_indexes(w, 'log_v_noise')] = np.log(1.0)
        self.weights = w
        self.nonlinearity = nonlinearity
        # Number of latent parameters per instance
        self.num_latent_params = param_set['weight_count']
        self.num_state_dims = param_set['num_state_dims']
        # Number of samples drawn from BNN parameter distributions
        self.num_weight_samples = param_set['bnn_num_samples']
        # Parameters for BNN training
        self.alpha = param_set['bnn_alpha']
        self.N = param_set['bnn_batch_size'] * param_set['num_strata_samples']
        self.bnn_batch_size = param_set['bnn_batch_size']
        self.train_epochs = param_set['bnn_training_epochs']
        self.v_prior = param_set['bnn_v_prior']
        self.learning_rate = param_set['bnn_learning_rate']
        # Parameters for latent weights
        if 'wb_num_epochs' in param_set:
            self.wb_opt_epochs = param_set['wb_num_epochs']
        else:
            self.wb_opt_epochs = 100
        # use separate learning rate for latent weights
        self.wb_learning_rate = param_set['wb_learning_rate']

    def __unpack_layers__(self, weight_samples):
        for m, n in self.shapes:
            yield weight_samples[:, :(m*n)].reshape((self.num_weight_samples, m, n)), \
                  weight_samples[:, (m*n):((m*n) + n)].reshape((self.num_weight_samples, 1, n))
            weight_samples = weight_samples[:, ((m+1) * n):]

    def __predict__(self, weight_samples, inputs):
        for W, b in self.__unpack_layers__(weight_samples):
            outputs = np.matmul(inputs, W) + b
            inputs = self.nonlinearity(outputs)

        # Split outputs into mean and log-variances
        mean_outputs = outputs[:, :, :2]
        log_var_outputs = outputs[:, :, 2:]
        #print(outputs.shape)
        return mean_outputs, log_var_outputs

    def __predict__abon(self, weight_samples, inputs):
        if self.linear_latent_weights:
            num_inputs = inputs.shape[0]
            # Separate off latent weights
            latent_weights = inputs[:,-self.num_latent_params:]
            inputs = inputs[:,:-self.num_latent_params]
        for W, b in self.__unpack_layers__(weight_samples):
            outputs = np.matmul(inputs,W) + b
            #print("size of W",W.shape)
            #print("size of input", inputs.shape)
            inputs = self.nonlinearity(outputs)
            #print("size of outputs", outputs.shape)
        if self.linear_latent_weights:
            # First get NxDxK output
            first_outputs = np.reshape(outputs,(self.num_weight_samples,num_inputs,self.num_state_dims,self.num_latent_params))
            # Multiply by latent weights to get next states
            outputs = np.einsum('mndk,nk->mnd', first_outputs, latent_weights)
        #print("outputs",outputs.size)
        return outputs

    def __log_likelihood_factor__(self, samples_q, v_noise, X, wb, y):
        # Account for occasions where we're optimizing the latent weighting distributions
        if wb.shape[0] == 1:
            if wb.shape[1] > self.num_latent_params:
                # Reshape the wb to be a full matrix and build full latent array
                Wb = np.reshape(wb, [-1, self.num_latent_params])
                latent_weights = np.array([Wb[int(X[tt, -1]), :] for tt in range(X.shape[0])])
                mean_outputs, log_var_outputs = self.__predict__(samples_q, np.hstack([X[:, :-1], latent_weights]))
            else:
                mean_outputs, log_var_outputs = self.__predict__(samples_q,
                                                                 np.hstack([X, np.tile(wb, (X.shape[0], 1))]))
        else:
            mean_outputs, log_var_outputs = self.__predict__(samples_q, np.hstack([X, wb]))

        # Compute the Gaussian log likelihood using the predicted means and variances
        var_outputs = np.exp(log_var_outputs)
        log_lik = -0.5 * (log_var_outputs + ((y - mean_outputs) ** 2) / var_outputs)
        # Return the likelihood in the shape of (2, 50, 32, 2)
        # [The '2' is just an arbitrary dimension, you'll need to adjust as per your specific needs]
        return np.expand_dims(log_lik, 0).repeat(2, axis=0)

    def __log_likelihood_factor__abon(self, samples_q, v_noise, X, wb, y):
        # Account for occasions where we're optimizing the latent weighting distributions
        if wb.shape[0] == 1:
            if wb.shape[1] > self.num_latent_params: # Further account
                # Reshape the wb to be a full matrix and build full latent array
                Wb = np.reshape(wb, [-1,self.num_latent_params])
                latent_weights = np.array([Wb[int(X[tt,-1]),:] for tt in range(X.shape[0])])
                outputs = self.__predict__(samples_q, np.hstack([X[:,:-1], latent_weights]))
            else:
                outputs = self.__predict__(samples_q, np.hstack([X, np.tile(wb,(X.shape[0],1))]))
        else:
            outputs = self.__predict__(samples_q, np.hstack([X, wb]))
        #print(((-0.5*np.log(2*math.pi*v_noise)) - (0.5*(np.tile(np.expand_dims(y,axis=0), (self.num_weight_samples,1,1))-outputs)**2)/v_noise).shape)
        return (-0.5*np.log(2*math.pi*v_noise)) - (0.5*(np.tile(np.expand_dims(y,axis=0), (self.num_weight_samples,1,1))-outputs)**2)/v_noise

    #def __draw_samples__(self, q):
    #    return npr.randn(self.num_weight_samples, len(q['m'])) * np.sqrt(q['v']) + q['m']

    def __draw_samples__(self, q, k=1):
        return npr.randn(self.num_weight_samples, len(q['m'])) * k * np.sqrt(q['v']) + q['m']

    '''
    def sample_from_tail(self, num_samples, stddev_limit=1):
        samples = npr.randn(num_samples)
        # Resample values that are within stddev_limit
        while np.any(np.abs(samples) < stddev_limit):
            mask = np.abs(samples) < stddev_limit
            samples[mask] = npr.randn(np.sum(mask))
        return samples

    def __draw_samples__(self, q, k=1):
        tail_samples = self.sample_from_tail(self.num_weight_samples * len(q['m']))
        tail_samples = tail_samples.reshape(self.num_weight_samples, len(q['m']))
        return tail_samples * np.sqrt(k * q['v']) + q['m']
    '''

    def __logistic__(self, x): return 1.0 / (1.0+np.exp(-x))

    def __get_parameters_q__(self, weights, scale=1.0):
        v = self.v_prior * self.__logistic__(self.parser.get(weights,'log_variance'))[:,0]
        #print("check v!!!!!:",v.shape)
        m = self.parser.get(weights,'mean')[:,0]
        return {'m': m, 'v': v}

    def __get_parameters_f_hat__(self, q):
        v = 1.0 / (1.0/self.N*(1.0/q['v']-1.0/self.v_prior))
        m = 1.0 / self.N * q['m'] / q['v'] * v
        return {'m': m, 'v': v}

    def __log_normalizer__(self, q):
        return np.sum(0.5*np.log(q['v']*2*math.pi) + 0.5*q['m']**2/q['v'])

    def __log_Z_prior__(self):
        return self.num_weights * 0.5 * np.log(self.v_prior*2*math.pi)

    def __log_Z_likelihood__(self, q, f_hat, v_noise, X, wb, y):
        samples = self.__draw_samples__(q)
        log_f_hat = np.sum(-0.5/f_hat['v']*samples**2 + f_hat['m']/f_hat['v']*samples, 1)
        log_factor_value = self.alpha * (np.sum(self.__log_likelihood_factor__(samples, v_noise, X, wb, y), axis=2) - np.expand_dims(log_f_hat,1))
        result = np.sum(logsumexp(log_factor_value, 0) + np.log(1.0 / self.num_weight_samples))
        return result

    def __make_batches__(self, shape=0):
        if shape > 0:
            return [slice(i, min(i+self.bnn_batch_size, shape)) for i in range(0, shape, self.bnn_batch_size)]
        else:
            return [slice(i, min(i+self.bnn_batch_size, self.N)) for i in range(0, self.N, self.bnn_batch_size)]

    def energy(self, weights,  X, wb, y):

        v_noise = np.exp(self.parser.get(weights, 'log_v_noise')[0,0])
        q = self.__get_parameters_q__(weights)
        f_hat = self.__get_parameters_f_hat__(q)
        result = -self.__log_normalizer__(q) - 1.0 * self.N / X.shape[0] / self.alpha * self.__log_Z_likelihood__(q, f_hat,
                                                                                                         v_noise, X, wb,y) + self.__log_Z_prior__()

        return result

    def get_error_and_ll(self, X, y, location, scale):
        v_noise = np.exp(self.parser.get(self.weights, 'log_v_noise')[0,0]) * scale**2
        q = self.__get_parameters_q__()
        samples_q = self.__draw_samples__(q)
        outputs, var = self.__predict__(samples_q, X)
        log_factor = -0.5*np.log(2*math.pi*v_noise) - 0.5*(np.tile(np.expand_dims(y,axis=0), (self.num_weight_samples,1,1))-np.array(outputs))**2/v_noise
        ll = np.mean(logsumexp(np.sum(log_factor,2)-np.log(self.num_weight_samples), 0))
        error = np.sqrt(np.mean((y-np.mean(outputs, axis=0))**2))
        return error, ll


    def feed_forward_ab(self, X, location=0.0, scale=1.0):
        q = self.__get_parameters_q__(self.weights)
        samples_q = self.__draw_samples__(q)
        all_samples_outputs, var = self.__predict__(samples_q, X)
        #print("output shape", all_samples_outputs.shape)\
        #print(all_samples_outputs.shape)
        outputs = np.mean(all_samples_outputs, axis = 0)*scale + location
        #outputs = np.round(outputs).astype(int)
        uncertainties = np.std(all_samples_outputs, axis=0)
        #print("uncertainties:",uncertainties)
        return outputs

    def feed_forward(self, X, location=0.0, scale=1.0):
        q = self.__get_parameters_q__(self.weights)
        samples_q = self.__draw_samples__(q)
        means, log_vars = self.__predict__(samples_q, X)
        #print(means.shape, log_vars.shape)
        # Assuming the last half of the outputs are log-variances
        #means = all_samples_outputs[:, :, :self._output_size]
        #log_vars = all_samples_outputs[:, :, self._output_size:]

        # You can average over means and log_vars if you're trying to get a point estimate
        avg_means = np.mean(means, axis=0) * scale + location
        avg_log_vars = np.mean(log_vars, axis=0)

        # Epistemic uncertainty can still be measured from the spread of means
        epistemic_uncertainties = np.std(means, axis=0)

        # Aleatoric uncertainty is directly from the predicted log-variance
        aleatoric_uncertainties = np.exp(avg_log_vars)  # converting log-variance to standard deviation

        return avg_means, avg_log_vars


    def feed_forward_distribution(self, X, location=0.0, scale=1.0):
        q = self.__get_parameters_q__(self.weights)
        samples_q = self.__draw_samples__(q)
        means, log_vars = self.__predict__(samples_q, X)
        #print(means.shape, log_vars.shape)
        # Assuming the last half of the outputs are log-variances
        #means = all_samples_outputs[:, :, :self._output_size]
        #log_vars = all_samples_outputs[:, :, self._output_size:]
        #print(means.shape)
        # You can average over means and log_vars if you're trying to get a point estimate
        avg_means = np.mean(means, axis=0) * scale + location
        #print(means.shape)
        avg_log_vars = np.mean(log_vars, axis=0)
        #aleatoric_uncertainties = np.std(log_vars, axis=0)
        #print("std log!!!",std_log_vars)
        # Epistemic uncertainty can still be measured from the spread of means
        epistemic_uncertainties = np.std(means, axis=0)
        # Aleatoric uncertainty is directly from the predicted log-variance
        aleatoric_uncertainties = np.exp(avg_log_vars)  # converting log-variance to standard deviation

        return avg_means, means, epistemic_uncertainties, aleatoric_uncertainties


    def get_td_error_ab(self, X, y, location=0.0, scale=1.0, by_dim=False):
        # Get the mean and variance predictions for each transition tuple in X
        predicted_means, predicted_log_vars = self.feed_forward(X, location, scale)

        # Convert the log variance to variance for ease of use
        predicted_variances = np.exp(predicted_log_vars)

        # Compute the error between predicted means and true values
        errors = (y - predicted_means)

        # Compute the weighted squared error, taking variance into account
        weighted_errors = errors ** 2 / (predicted_variances + 1e-6)  # small epsilon to avoid division by zero

        if by_dim:
            return weighted_errors
        return np.sqrt(np.sum(weighted_errors, axis=1))

    def get_td_error(self, X, y, location=0.0, scale=1.0, by_dim=False):
        # Compute the L2 norm of the error for each transition tuple in X
        outputs, predicted_log_vars = self.feed_forward(X, location, scale)
        #print(outputs, len(outputs))
        if by_dim:
            return (y-outputs)**2
        return np.sqrt(np.sum((y-outputs)**2, axis=1))

    def get_prediction_std(self, X, location=0.0, scale=1.0):
        q = self.__get_parameters_q__(self.weights)
        samples_q = self.__draw_samples__(q)
        all_samples_outputs, var = self.__predict__(samples_q, X)*scale + location
        std_by_dim = np.std(all_samples_outputs, axis = 0)
        return np.sum(std_by_dim)


    def fit_network(self, exp_buffer, task_weights, task_steps, state_diffs=False, use_all_exp=False):
        """Learn BNN network weights using gradients of the energy function with respect to the network weights
        and performing minibatch updates via SGD (ADAM).

        Arguments:
        exp_buffer -- Either an ExperienceReplay object, or a list of transitions;
            if use_all_exp==False, then an ExperienceReplay object must be supplied;
            otherwise, a list of transitions must be supplied (where each transition is a numpy array)
        task_weights -- the latent weights: a numpy array of with dimensions (number of instances x number of latent weights)
        task_steps --  total steps taken in environment

        Keyword Arguments:
        state_diffs -- boolean indicating if the BNN should predict state differences rather than the next state (default: False)
        use_all_exp -- boolean indicating whether updates should be performed using all experiences
        """
        # Create gradient functional of the energy function wrt W
        energy_grad = grad(self.energy, argnum=0)
        weights = np.copy(self.weights)
        m1 = 0
        m2 = 0
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t = 0

        for epoch in range(self.train_epochs):
            # Gather a sample of data from the experience buffer, convert to input and target arrays
            if use_all_exp:
                batch = exp_buffer
            else:
                batch, __, indices = exp_buffer.sample(task_steps)
            X = np.array([np.hstack([batch[tt,0], batch[tt,1]]) for tt in range(len(batch))])
            wb = np.array([task_weights[batch[tt,4],:] for tt in range(len(batch))])
            y = np.array([batch[tt,3] for tt in range(len(batch))])
            if state_diffs:
                y = y - X[:, :batch[0,0].shape[0]]
            self.N = X.shape[0]
            batch_idxs = self.__make_batches__()
            # Permute the indices of the training inputs for SGD purposes
            permutation = np.random.permutation(X.shape[0])
            for idxs in batch_idxs:
                t += 1
                grad_w = energy_grad(weights, X[permutation[idxs]], wb[permutation[idxs]], y[permutation[idxs]])
                m1 = beta1*m1 + (1-beta1)*grad_w
                m2 = beta2*m2 + (1-beta2)*grad_w**2
                m1_hat = m1 / (1-beta1**t)
                m2_hat = m2 / (1-beta2**t)
                weights = weights - self.learning_rate*m1_hat/(np.sqrt(m2_hat)+epsilon)
            # Re-queue sampled data with updated TD-error calculations
            self.weights = weights
            if (not use_all_exp) and exp_buffer.mem_priority:
                td_loss = self.get_td_error(np.hstack([X, wb]), y, 0.0, 1.0)
                exp_buffer.update_priorities(np.hstack((np.reshape(td_loss,(len(td_loss),-1)), np.reshape(indices,(len(indices),-1)))))

    def optimize_latent_weighting_stochastic(self, exp_buffer, wb, task_steps, state_diffs=False, use_all_exp=False):
        """Learn the latent weights using gradients of the energy function with respect to the latent weights
        and performing minibatch updates via SGD (ADAM).

        Arguments:
        exp_buffer -- Either an ExperienceReplay object, or a list of transitions of a single instance;
            if use_all_exp==False, then an ExperienceReplay object must be supplied;
            otherwise, a list of transitions must be supplied (where each transition is a numpy array)
        wb -- the latent weights for the specific instance
        task_steps --  total steps taken in environment

        Keyword Arguments:
        state_diffs -- boolean indicating if the BNN should predict state differences rather than the next state (default: False)
        use_all_exp -- boolean indicating whether updates should be performed using all experiences
        """
        # Create gradient functional of the energy function wrt wb
        energy_grad = grad(self.energy, argnum=2)
        cur_latent_weights = wb
        m1 = 0
        m2 = 0
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t = 0
        # With linear top latent weights, use a single sample of the BNN network weights to compute gradients
        if self.linear_latent_weights:
            tmp_num_weight_samples = self.num_weight_samples
            self.num_weight_samples = 1
        for epoch in range(self.wb_opt_epochs):
            # Gather a sample of data from the experience buffer, convert to input and target arrays
            if use_all_exp:
                batch = exp_buffer
            else:
                batch, __, indices = exp_buffer.sample(task_steps)
            X = np.array([np.hstack([batch[tt,0], batch[tt,1]]) for tt in range(len(batch))])
            y = np.array([batch[tt,3] for tt in range(len(batch))])
            if state_diffs:
                y = y - X[:, :batch[0,0].shape[0]]
            self.N = X.shape[0]
            batch_idxs = self.__make_batches__()
            # Permute the indices of the training inputs for SGD purposes
            #permutation = np.random.permutation(X.shape[0])
            permutation = np.random.choice( range(X.shape[0]), X.shape[0], replace=False )
            for idxs in batch_idxs:
                t += 1
                #print(cur_latent_weights)
                grad_wb = energy_grad(self.weights, X[permutation[idxs]], cur_latent_weights, y[permutation[idxs]])
                m1 = beta1*m1 + (1-beta1)*grad_wb
                m2 = beta2*m2 + (1-beta2)*grad_wb**2
                m1_hat = m1 / (1-beta1**t)
                m2_hat = m2 / (1-beta2**t)
                cur_latent_weights -= self.wb_learning_rate * m1_hat / (np.sqrt(m2_hat)+epsilon)
                #print("checkpoint inside3")
            # Re-queue sampled data with updated TD-error calculations
            X_latent_weights = np.vstack([cur_latent_weights[0] for i in range(X.shape[0])])
            if not use_all_exp and exp_buffer.mem_priority:
                td_loss = self.get_td_error(np.hstack([X, X_latent_weights]), y, 0.0, 1.0)
                exp_buffer.update_priorities(np.hstack((np.reshape(td_loss,(len(td_loss),-1)), np.reshape(indices,(len(indices),-1)))))
        if self.linear_latent_weights:
            self.num_weight_samples = tmp_num_weight_samples
        return cur_latent_weights
