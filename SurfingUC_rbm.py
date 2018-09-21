# ------------------------------------------------------------------- #
# This Python code samples the sample space $\chi$ of the normalized  #
# maximum likelihood (NML) for the restricted Boltzmann machine (RBM) #
# which is used to plot Figure 4 in the text:                         #
#    Minimum Description Length codes are critical                    #
#    Cubero, RJ; Marsili, M; Roudi, Y                                 #
#                                                                     #
# WARNING: This code may take a long time to converge. Proceed to     #
# increase the size of the system at your own risk.                   #
# ------------------------------------------------------------------- #

# Some basic imports
from __future__ import division
import numpy as np

# Import important libraries
from collections import Counter
from multiprocessing import Pool

# Import functions from an external library
from spin_operators import *
from boltzmann_learning import *
from newer_RBM import *
from relevance import *

# Define the rng seed for reproducibility
np.random.seed(34151)

# Define the size $M$ of sample $\hat{s}$ and the number of variables in the visible layer
M = 1000
N_v = 10

def surf_rbm(N_h):
    output_prefix = 'UC_RBM_Nv%s_Nh%s_M%s'%(str(N_v), str(N_h), str(M))
    
    arguments = {}
    configs = np.array([binary_array(mu, N_v) for mu in np.arange(np.power(2,N_v))])
    arguments['configs'] = configs
    arguments['M'] = M
    arguments['N_v'] = N_v
    arguments['N_h'] = N_h
    
    # Initialize the RBM and generate a finite-size sample
    Wtrue = 4*np.random.randn(N_v+1, N_h+1)/np.sqrt(N_v+1)
    x = np.ones((N_v)).astype('int')
    x = (np.random.rand(N_v) >= 0.5).astype('int')
    initial_data = sample_RBM(Wtrue, M, 1000, 100, x)
    initial_data = initial_data.astype('int')
    
    # Infer the RBM from the generated sample by CD-10
    arguments['data'] = initial_data
    r = RBM(num_visible = N_v, num_hidden = N_h, initial_W = Wtrue)
    r.train(initial_data, num_epochs = 2500, learning_rate=0.01, CD_steps=10, momentum=0.5, batch_size=200)
    initial_loglikelihood = -RBM_loglikelihood(r.weights, arg_data=arguments)[0]/M
    initial_weights = r.weights.copy()
    
    uc_samples = []
    free_energy = []
    HofK = []
    HofS = []
    accept = []
    iterations = 0
    
    # Generate 100 samples from the NML code of the RBM
    while len(HofK)<=100:
        test_data = np.copy(initial_data)
        
        # Doing $n_flips = M$ ensures fast mixing, not necessarily proper however, at some point in the Markov chain Monte Carlo,
        # we flip 10 spins as we find that this is a good trade-off between the acceptance ratio and the running time of the code
        if iterations<500:
            n_flips = M
            n_epochs = 2500
        else:
            n_flips = 10
            n_epochs = 1000
        
        # Flipping: randomly choose $n_flips$ configurations and randomly flip a spin within this configuration
        # This creates a new sample
        rand_state = np.random.randint(M, size=n_flips)
        rand_pos = np.random.randint(N_v, size=n_flips).astype('int')
        for k in np.arange(n_flips):
            test_data[rand_state[k]][rand_pos[k]] = int(~(test_data[rand_state[k]][rand_pos[k]]).astype('bool'))
    
        # Use the previously inferred couplings to infer the model
        # When $n_flips$ is small, the model to be inferred should not be "too far" from the previously inferred model
        arguments['data'] = test_data
        r = RBM(num_visible = N_v, num_hidden = N_h, initial_W = initial_weights)
        r.train(test_data, num_epochs = n_epochs, learning_rate=0.01, CD_steps=10, momentum=0.5, batch_size=200)
        test_loglikelihood = -RBM_loglikelihood(r.weights, arg_data=arguments)[0]/M
        
        dg = test_loglikelihood - initial_loglikelihood
        if iterations<200: condition = np.exp(dg)
        else: condition = np.exp(M*dg)
        random_number = np.random.rand()
        
        if (np.min([condition,1]) >= random_number):
            initial_data = np.copy(test_data)
            initial_loglikelihood = np.copy(test_loglikelihood)
            initial_weights = r.weights.copy()
            accept.append(1)
        
        else:
            accept.append(0)
    
        # Equilibrate the Markov chain Monte Carlo at 1000 steps before calculating relevant quantities
        if ((np.mod(iterations,100)==0) and (iterations>=(1000))):
            initial_data_index = b2i_states(initial_data)
            HK, HS = calculate_HofKS(np.array(list(Counter(initial_data_index).values())))
            HofK.append(HK); HofS.append(HS)
            free_energy.append(initial_loglikelihood)
            uc_samples.append(initial_data_index)
            print(iterations, accept[-1], np.mean(accept), HK, HS)
            accept = []

        iterations += 1
            
    np.savetxt(output_prefix+"_relevance.d", np.array(HofK))
    np.savetxt(output_prefix+"_resolution.d", np.array(HofS))
    np.savetxt(output_prefix+"_loglikelihood.d", np.array(free_energy))
    np.savetxt(output_prefix+"_samples.d", np.array(uc_samples))

# Iterate over a range of number of hidden spins, $n_h$, in the system
pool = Pool()
res = pool.map_async(surf_rbm,np.arange(3,8,1))
pool.close(); pool.join()
