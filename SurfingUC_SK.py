# ------------------------------------------------------------------- #
# This Python code samples the sample space $\chi$ of the normalized  #
# maximum likelihood (NML) for the Sherrington-Kirkpatrick model      #
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
from relevance import *

# Define the size $M$ of sample $\hat{s}$
M = 1000

# Define a function for surfing the sample space of the NML codes for the SK model as a function of the size, $N$ of the sample
def surf_ising(N):
    output_name = 'UC_SK_N'+str(N)+'_'
    # Define the spin operators and the observables
    ising_operators = np.array([i for i in np.arange(np.power(2,N)) if (binary(i,N).count('1')==1 or binary(i,N).count('1')==2)])
    single_spin = np.array([i for i in np.arange(np.power(2,N)) if binary(i,N).count('1')==1 ])
    phi_matrix = np.array([ [ phi(mu, s, N) for s in np.arange(np.power(2,N)) ] for mu in ising_operators ] )

    # Initialize the SK model and calculate for the partition function and the corresponding probabilities
    # Since the number of spins in the SK model is small enough, we can enumerate the probabilities of the spin configurations
    pairwise_coup = np.random.normal(0,1./N,size=len(ising_operators))
    partition_function = np.sum(np.exp(np.dot(phi_matrix.T, pairwise_coup)))
    probabilities = np.exp(np.dot(phi_matrix.T, pairwise_coup))/partition_function
    
    # Sample the initialized SK model to get a finite-sized sample
    initial_data_index = np.random.choice(np.arange(np.power(2,N)),p=probabilities, size=M)

    # Count the frequency and averages of single and pairwise spin configurations needed for inference
    data_counts = Counter(initial_data_index)
    data_counts = [data_counts[s] if s in data_counts.keys() else 0 for s in np.arange(len(probabilities))]
    initial_averages = np.dot(phi_matrix, data_counts)/M

    # Calculate the correlation and the mean matrices needed for inference
    correlation_matrix = np.zeros((len(single_spin),len(single_spin)))
    for u,v in combinations(np.arange(len(single_spin)), 2):
        correlation_matrix[u][v] = initial_averages[np.where(ising_operators==((single_spin[u])^(single_spin[v])))[0][0]] - initial_averages[np.where(ising_operators==single_spin[u])[0][0]]*initial_averages[np.where(ising_operators==single_spin[v])[0][0]]
    mean_matrix = np.array([ initial_averages[np.where(ising_operators==single_spin[u])[0][0]] for u in np.arange(len(single_spin)) ])
    correlation_matrix = correlation_matrix + np.transpose(correlation_matrix) + np.diag(1.-np.power(mean_matrix,2))

    # Calculate the mean field values just to place the inferred model close to the maximum likelihood
    nMF_coupling = -np.linalg.inv(correlation_matrix) + np.diag(1./(1.-np.power(mean_matrix,2)))
    nMF_coupling = nMF_coupling - np.diag(np.diag(nMF_coupling))
    g0 = np.zeros(len(ising_operators))
    for u,v in combinations(np.arange(len(single_spin)), 2):
        g0[np.where(ising_operators==((single_spin[u])^(single_spin[v])))[0][0]] = nMF_coupling[u][v]

    # Infer the model which maximizes the likelihood by gradient descent
    initial_coup = infer(initial_averages, phi_matrix, N, g0)
    partition_function = np.sum(np.exp(np.dot(phi_matrix.T, initial_coup)))
    initial_Z = -np.log(partition_function)

    uc_samples = []
    energy = []
    loglikelihood = []
    HofK = []
    HofS = []
    accept = []
    iterations = 0

    # Generate 100 samples from the NML code of the SK model
    while len(HofK)<=100:
        # Doing $n_flips = M$ ensures fast mixing, not necessarily proper however, at some point in the Markov chain Monte Carlo,
        # we flip 10 spins as we find that this is a good trade-off between the acceptance ratio and the running time of the code
        if iterations<500: n_flips = M
        else: n_flips = 10

        test_data_index = initial_data_index.copy()
        
        # Flipping: randomly choose $n_flips$ configurations and randomly flip a spin within this configuration
        # This creates a new sample
        rand_state = np.random.randint(M, size=n_flips)
        rand_pos = np.random.randint(N, size=n_flips)
        for k in np.arange(n_flips):
            test_data_index[rand_state[k]] = spin_reversal((test_data_index[rand_state[k]], N, rand_pos[k]))

        # Calculate relevant quantities needed for inference
        data_counts = Counter(test_data_index)
        data_counts = [data_counts[s] if s in data_counts.keys() else 0 for s in np.arange(len(probabilities))]
        test_averages = np.dot(phi_matrix, data_counts)/M

        # Use the previously inferred couplings to infer the model
        # When $n_flips$ is small, the model to be inferred should not be "too far" from the previously inferred model
        test_coup = infer(test_averages, phi_matrix, N, initial_coup)
        partition_function = np.sum(np.exp(np.dot(phi_matrix.T, test_coup)))
        test_Z = -np.log(partition_function)

        dg = test_Z + np.dot(test_coup,test_averages) - initial_Z - np.dot(initial_coup,initial_averages)

        if iterations<500: condition = np.exp(dg)
        else: condition = np.exp(M*dg)
        random_number = np.random.rand()
        
        if (np.min([condition,1]) >= random_number):
            initial_Z = test_Z
            initial_data_index = test_data_index.copy()
            initial_averages = test_averages.copy()
            initial_coup = test_coup.copy()
            accept.append(1)
        else:
            accept.append(0)

        # Equilibrate the Markov chain Monte Carlo at 7000*N before calculating relevant quantities
        if ((np.mod(iterations,3000*N)==0) and (iterations>=7000*N)):
            energy.append(np.dot(initial_coup,initial_averages))
            loglikelihood.append(initial_Z + np.dot(initial_coup,initial_averages))
            HK, HS = calculate_HofKS(np.array(list(Counter(initial_data_index).values())))
            HofK.append(HK); HofS.append(HS)
            uc_samples.append(initial_data_index)
            print(N, iterations, accept[-1], np.mean(accept), HK, HS)
            accept = []

        iterations += 1

    np.savetxt(output_name+"energy.d", np.array(energy))
    np.savetxt(output_name+"loglikelihood.d", np.array(loglikelihood))
    np.savetxt(output_name+"relevance.d", np.array(HofK))
    np.savetxt(output_name+"resolution.d", np.array(HofS))
    np.savetxt(output_name+"samples.d", np.array(uc_samples))

# Iterate over a range of number of spins, $n$, in the system
pool = Pool()
res = pool.map_async(surf_ising,np.arange(3,10,1))
pool.close(); pool.join()
