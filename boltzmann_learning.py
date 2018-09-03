from __future__ import division

import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool
from itertools import combinations

from spin_operators import *

# defines the likelihoods.
def log_lik(couplings, averages, phi_matrix, N):
    assert len(couplings) == len(averages)
    
    # partition function
    Z = np.sum(np.exp(np.dot(phi_matrix.T, couplings)))
    
    # energy
    E = np.dot(averages,couplings)
    
    return E - np.log(Z)

# inference of best couplings
def infer(averages, phi_matrix, N, nMF_couplings, options={'gtol': 1e-5, 'disp': False}):
    def target(g):
        return -log_lik(g, averages, phi_matrix, N)
    
    # calculates result
    res = minimize(target, nMF_couplings, method='BFGS',options=options)
    g = res.x
    
    return g

def nMF_inference(single_spin, ising_operators, averages):
    correlation_matrix = np.zeros((len(single_spin),len(single_spin)))
    for u,v in combinations(np.arange(len(single_spin)), 2):
        correlation_matrix[u][v] = test_averages[np.where(ising_operators==((single_spin[u])^(single_spin[v])))[0][0]] - test_averages[np.where(ising_operators==single_spin[u])[0][0]]*test_averages[np.where(ising_operators==single_spin[v])[0][0]]
    mean_matrix = np.array([ test_averages[np.where(ising_operators==single_spin[u])[0][0]] for u in np.arange(len(single_spin)) ])
    correlation_matrix = correlation_matrix + np.transpose(correlation_matrix) + np.diag(1.-np.power(mean_matrix,2))
    nMF_coupling = -np.linalg.inv(correlation_matrix) + np.diag(1./(1.-np.power(mean_matrix,2)))
    nMF_coupling = nMF_coupling - np.diag(np.diag(nMF_coupling))
    nMF_fields = np.arctanh(mean_matrix) - np.dot(nMF_coupling,mean_matrix)

    g0 = np.zeros(len(ising_operators))
    for u,v in combinations(np.arange(len(single_spin)), 2):
        g0[np.where(ising_operators==((single_spin[u])^(single_spin[v])))[0][0]] = nMF_coupling[u][v]
    for u in np.arange(len(single_spin)):
        g0[np.where(ising_operators==single_spin[u])[0][0]] = nMF_fields[u]

    return g0
