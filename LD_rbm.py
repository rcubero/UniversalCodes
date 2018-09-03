# Some basic imports
from __future__ import division

import numpy as np

from collections import Counter
from multiprocessing import Pool

from spin_operators import *
from boltzmann_learning import *
from newer_RBM import *
from relevance import *

np.random.seed(34151)

M = 1000
N_v = 3
N_h = 3

def surf_ld_rbm(beta):
    output_prefix = 'UC_RBM_Nv%s_Nh%s_M%s_beta%s'%(str(N_v), str(N_h), str(M), str(100*beta))
    
    arguments = {}
    configs = np.array([binary_array(mu, N_v) for mu in np.arange(np.power(2,N_v))])
    arguments['configs'] = configs
    arguments['M'] = M
    arguments['N_v'] = N_v
    arguments['N_h'] = N_h
    
    Wtrue = 4*np.random.randn(N_v+1, N_h+1)/np.sqrt(N_v+1)
    x = np.ones((N_v)).astype('int')
    x = (np.random.rand(N_v) >= 0.5).astype('int')
    initial_data = sample_RBM(Wtrue, M, 1000, 100, x)
    initial_data = initial_data.astype('int')
    
    initial_data_index = b2i_states(initial_data)
    _, initial_HS = calculate_HofKS(np.array(list(Counter(initial_data_index).values())))
    
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
    
    while len(HofK)<=100:
        test_data = np.copy(initial_data)
        
        if iterations<500:
            n_flips = M
            n_epochs = 2500
        else:
            n_flips = 10
            n_epochs = 1000
        
        rand_state = np.random.randint(M, size=n_flips)
        rand_pos = np.random.randint(N_v, size=n_flips).astype('int')
        for k in np.arange(n_flips):
            test_data[rand_state[k]][rand_pos[k]] = int(~(test_data[rand_state[k]][rand_pos[k]]).astype('bool'))
    
        arguments['data'] = test_data
        r = RBM(num_visible = N_v, num_hidden = N_h, initial_W = initial_weights)
        
        r.train(test_data, num_epochs = n_epochs, learning_rate=0.01, CD_steps=10, momentum=0.5, batch_size=200)
        test_loglikelihood = -RBM_loglikelihood(r.weights, arg_data=arguments)[0]/M
        
        test_data_index = b2i_states(test_data)
        _, test_HS = calculate_HofKS(np.array(list(Counter(test_data_index).values())))
        
        dg = test_loglikelihood - beta*test_HS - initial_loglikelihood + beta*initial_HS
        if iterations<200: condition = np.exp(dg)
        else: condition = np.exp(M*dg)
        random_number = np.random.rand()
        
        if (np.min([condition,1]) >= random_number):
            initial_data = test_data.copy()
            initial_loglikelihood = np.copy(test_loglikelihood)
            initial_weights = r.weights.copy()
            initial_HofS = test_HS
            initial_data_index = test_data_index
            accept.append(1)
        
        else:
            accept.append(0)
    
        if ((np.mod(iterations,100)==0) and (iterations>=(1000))):
            HofS.append(initial_HS)
            uc_samples.append(initial_data_index)
            print(iterations, accept[-1], np.mean(accept), initial_HS)
            accept = []

        iterations += 1
            
    np.savetxt(output_prefix+"_resolution.d", np.array(HofS))
    np.savetxt(output_prefix+"_samples.d", np.array(uc_samples))

pool = Pool()
res = pool.map_async(surf_ld_rbm,[-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
pool.close(); pool.join()

