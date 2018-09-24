# ------------------------------------------------------------------- #
# This Python code samples the large deviation realizations for the   #
# normalized maximum likelihood (NML) of the Sherrington-Kirkpatrick  #
# model. This is rather a supplementary code and does not appear in   #
# the text:                                                           #
#    Minimum Description Length codes are critical                    #
#    Cubero, RJ; Marsili, M; Roudi, Y                                 #
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
from relevance import *

# Define the size $M$ of sample $\hat{s}$ and the size, $N$ of the system
M = 1000
N = 3

# define a function which realizes large deviation samples in the coding cost as a function of the parameter beta
def surf_ld_ising(beta):
    output_name = 'UC_SK_N'+str(N)+'_b'+str(100*beta)+'_'
    ising_operators = np.array([i for i in np.arange(np.power(2,N)) if (binary(i,N).count('1')==1 or binary(i,N).count('1')==2)])
    single_spin = np.array([i for i in np.arange(np.power(2,N)) if binary(i,N).count('1')==1 ])
    phi_matrix = np.array([ [ phi(mu, s, N) for s in np.arange(np.power(2,N)) ] for mu in ising_operators ] )

    pairwise_coup = np.random.normal(0,1./N,size=len(ising_operators))
    partition_function = np.sum(np.exp(np.dot(phi_matrix.T, pairwise_coup)))
    probabilities = np.exp(np.dot(phi_matrix.T, pairwise_coup))/partition_function
    initial_data_index = np.random.choice(np.arange(np.power(2,N)),p=probabilities, size=M)

    data_counts = Counter(initial_data_index)
    data_counts = [data_counts[s] if s in data_counts.keys() else 0 for s in np.arange(len(probabilities))]
    initial_averages = np.dot(phi_matrix, data_counts)/M

    correlation_matrix = np.zeros((len(single_spin),len(single_spin)))
    for u,v in combinations(np.arange(len(single_spin)), 2):
        correlation_matrix[u][v] = initial_averages[np.where(ising_operators==((single_spin[u])^(single_spin[v])))[0][0]] - initial_averages[np.where(ising_operators==single_spin[u])[0][0]]*initial_averages[np.where(ising_operators==single_spin[v])[0][0]]
    mean_matrix = np.array([ initial_averages[np.where(ising_operators==single_spin[u])[0][0]] for u in np.arange(len(single_spin)) ])
    correlation_matrix = correlation_matrix + np.transpose(correlation_matrix) + np.diag(1.-np.power(mean_matrix,2))
    nMF_coupling = -np.linalg.inv(correlation_matrix) + np.diag(1./(1.-np.power(mean_matrix,2)))
    nMF_coupling = nMF_coupling - np.diag(np.diag(nMF_coupling))
    g0 = np.zeros(len(ising_operators))
    for u,v in combinations(np.arange(len(single_spin)), 2):
        g0[np.where(ising_operators==((single_spin[u])^(single_spin[v])))[0][0]] = nMF_coupling[u][v]
    
    initial_coup = infer(initial_averages, phi_matrix, N, g0)
    partition_function = np.sum(np.exp(np.dot(phi_matrix.T, initial_coup)))
    initial_Z = -np.log(partition_function)
    _, initial_HS = calculate_HofKS(np.array(list(Counter(initial_data_index).values())))

    uc_samples = []
    energy = []
    loglikelihood = []
    HofK = []
    HofS = []
    accept = []
    iterations = 0

    while len(HofK)<=100:
        if iterations<500: n_flips = M
        else: n_flips = 1

        test_data_index = initial_data_index.copy()
        
        rand_state = np.random.randint(M, size=n_flips)
        rand_pos = np.random.randint(N, size=n_flips)

        for k in np.arange(n_flips):
            test_data_index[rand_state[k]] = spin_reversal((test_data_index[rand_state[k]], N, rand_pos[k]))

        data_counts = Counter(test_data_index)
        data_counts = [data_counts[s] if s in data_counts.keys() else 0 for s in np.arange(len(probabilities))]
        test_averages = np.dot(phi_matrix, data_counts)/M

        test_coup = infer(test_averages, phi_matrix, N, initial_coup)
        partition_function = np.sum(np.exp(np.dot(phi_matrix.T, test_coup)))
        test_Z = -np.log(partition_function)
        _, test_HS = calculate_HofKS(np.array(list(Counter(test_data_index).values())))

        dg = test_Z + np.dot(test_coup,test_averages) - beta*M*test_HS - initial_Z - np.dot(initial_coup,initial_averages) + beta*M*initial_HS

        if iterations<500: condition = np.exp(dg)
        else: condition = np.exp(M*dg)
        random_number = np.random.rand()
        
        if (np.min([condition,1]) >= random_number):
            initial_Z = test_Z
            initial_data_index = test_data_index.copy()
            initial_averages = test_averages.copy()
            initial_coup = test_coup.copy()
            initial_HS = test_HS
            accept.append(1)
        else:
            accept.append(0)
    
        if ((np.mod(iterations,3000*N)==0) and (iterations>=5000*N)):
            HofS.append(initial_HS)
            uc_samples.append(initial_data_index)
            print(beta, iterations, accept[-1], np.mean(accept), initial_HS)
            accept = []

        iterations += 1

    np.savetxt(output_name+"resolution.d", np.array(HofS))
    np.savetxt(output_name+"samples.d", np.array(uc_samples))

#pool = Pool()
#res = pool.map_async(surf_ld_ising,[-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
#pool.close(); pool.join()

surf_ld_ising(-2.0)

