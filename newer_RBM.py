from scipy.optimize import minimize
from multiprocessing import Pool

from spin_operators import *

def sigm(X):
    return 1./(1.+np.exp(-X))

class RBM():
    def __init__(self, num_hidden, num_visible, initial_W):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        
        self.weights = initial_W.copy()
        self.delta_weights = np.zeros_like(self.weights)
    
    def train(self, data, num_epochs=1000, learning_rate=0.1, momentum = 0.5, CD_steps=1, batch_size=100):
        self.momentum = momentum
        self.learning_rate = learning_rate
        
        #Add a bias column
        data = np.insert(data, 0, 1, axis = 1)
        
        num_samples, num_visible = data.shape
        
        for n in range(num_epochs):
            subsample = data[np.random.choice(num_samples, batch_size)]
            visible_sample = subsample.copy()
            
            #Make CD steps of Gibbs samples
            hidden_activations, hidden_sample, visible_activations, visible_sample = self.gibbs_sample(visible_sample, CD_steps)
            
            # Now calculate both parts of eq.6 in PG_RBM
            E_data = np.dot(subsample.T, hidden_activations )
            E_model = np.dot(visible_activations.T, hidden_activations)
            
            # Update the weights
            gradient = (E_data - E_model) / batch_size
            self.update_weights(gradient, n)

    def gibbs_sample(self, visible_sample, steps=3):
        """
        Performs block Gibbs sampling. Alternatively samples the hidden states
        and the visible states
        :param visible_sample:
        :param steps:
        :return: None
        """
        batch_size = visible_sample.shape[0]
        for step in range(steps):
            # Updating the hidden states
            # Section 3.1 PG_RBM
            hidden_activations = sigm(np.dot(visible_sample, self.weights))
            hidden_activations[:,0] = 1. #Biases are always 1
            hidden_sample = hidden_activations > np.random.rand(batch_size, self.num_hidden + 1)
            
            # Updating the visible states
            # Section 3.2 PG_RBM
            visible_activations = sigm(np.dot(hidden_sample, self.weights.T))
            visible_activations[:,0] = 1. #Biases are always 1
            visible_sample = visible_activations > np.random.rand(batch_size, self.num_visible + 1)
        return hidden_activations, hidden_sample, visible_activations, visible_sample


    def update_weights(self, gradient, n):
        """
        Updates the weights according to the gradient. Implementing both learning rate decay
        and momentum
        :param gradient: the gradient wrt the weights
        :param n: step
        :return: None
        """
        lr_decay = self.learning_rate*(1/2)*(n/100)
        self.delta_weights = lr_decay * gradient + self.momentum*self.delta_weights
        self.weights += self.delta_weights


    def generate_samples(self, num_samples = 25):
        visible_sample = np.random.rand(1,785)
        visible_sample[0,0] = 1
        samples = []
        for n in range(num_samples):
            hidden_activations, hidden_sample, visible_activations, visible_sample = self.gibbs_sample(visible_sample, 10)
            samples.append(visible_activations)
        return samples


def calc_RBM_partition(zipped_data):
    weights, config = zipped_data
    return np.sum(np.log(1+np.exp(np.dot(np.insert(config, 0, 1), weights[:,1:])))) + np.dot(np.insert(config, 0, 1), weights[:,0])

def RBM_loglikelihood(g, arg_data):
    data = arg_data['data']
    configs = arg_data['configs']
    M = arg_data['M']
    N_v = arg_data['N_v']
    N_h = arg_data['N_h']
    
    g = g.reshape(N_v+1, N_h+1)
    dg = np.zeros_like(g)
    
    J = g[1:,1:]
    b = g[1:,0]
    c = g[0,1:]
    
    # calculate partition function
    exp_arg = np.dot(configs, J) + c
    exp_arg = np.dot(configs, b) + np.sum(np.log(1.+np.exp(exp_arg)),axis=1) + g[0,0]
    Z = np.sum(np.exp(exp_arg))
    
    # calculate log-likelihood
    exp_arg = np.dot(data, J) + c
    exp_arg = np.dot(data, b) + np.sum(np.log(1.+np.exp(exp_arg)),axis=1) + g[0,0]
    L = np.sum(exp_arg)-M*np.log(Z)
    
    # calculate gradient
    exp_arg = np.dot(data, J) + c
    dg[1:,1:] = np.dot(data.T, np.exp(exp_arg)/(1. + np.exp(exp_arg)))
    dg[0,1:] = np.sum(np.exp(exp_arg)/(1. + np.exp(exp_arg)), axis=0)
    dg[1:,0] = np.sum(data, axis=0)
    
    dg = dg.flatten()
    
    return -L, -dg

def infer_RBM(weights, data, configs, options={'gtol': 1e-5, 'disp': False}):
    M = len(data)
    N_v, N_h = weights.shape
    N_v -= 1; N_h -= 1; # remove bias terms
    
    arguments = {}
    arguments['data'] = data
    arguments['configs'] = configs
    arguments['M'] = M
    arguments['N_v'] = N_v
    arguments['N_h'] = N_h
    
    res = minimize(RBM_loglikelihood, weights.flatten(), method='L-BFGS-B', jac=True, options=options, args=arguments)
    g = res.x
    g = g.reshape(N_v+1, N_h+1)
    
    return g

def sample_RBM(W, nsamples, burnin, independent_steps, x):
    nsamplingsteps = burnin + 1 + (nsamples-1)*(independent_steps)
    ndims, nhid = W.shape
    nhid -= 1; ndims -= 1;
    
    X_out = np.zeros((nsamples, ndims))
    visible_activations = np.ones(ndims+1)
    visible_activations[1:] = x
    visible_activations[0] = 1 #bias unit
    
    i_out = 1
    next_sample = burnin+1
    
    for si in np.arange(nsamplingsteps):
        hidden_activations = sigm(np.dot(visible_activations, W))
        hidden_activations[0] = 1. #Biases are always 1
        hidden_activations = (hidden_activations > np.random.rand(nhid + 1)).astype('int')
        
        visible_activations = sigm(np.dot(hidden_activations, W.T))
        visible_activations[0] = 1. #Biases are always 1
        visible_activations = (visible_activations > np.random.rand(ndims + 1)).astype('int')
        
        if si == next_sample:
            next_sample = si + independent_steps
            X_out[i_out] = visible_activations[1:]
            i_out += 1

    return X_out
