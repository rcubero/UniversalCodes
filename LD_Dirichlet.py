# ------------------------------------------------------------------- #
# This Python code samples the large deviation realizations of the    #
# normalized maximum likelihood (NML) for the Dirichlet model which   #
# is used to plot Figure 5 in the text:                               #
#    Minimum Description Length codes are critical                    #
#    Cubero, RJ; Marsili, M; Roudi, Y                                 #
# ------------------------------------------------------------------- #

# Some basic imports
from __future__ import division
import numpy as np

# Import necessary libraries
from collections import Counter
from mpmath import mp, log, loggamma, exp, fmul, fsum, power

# Configure mpmath to calculate quantities up to 100 decimal places
mp.dps = 100
mp.pretty = True

# Define function to calculate $\hat{H}[k]$ and $\hat{H}[s]$
def calculate_HofKS(mapping_ks):
    ks_counts = np.asarray(Counter(mapping_ks).most_common())
    positive_values = np.where(ks_counts[:,0]>0)[0]
    kq, mq = ks_counts[:,0][positive_values], ks_counts[:,1][positive_values]
    assert np.sum(kq*mq)==np.sum(mapping_ks)
    M = float(np.sum(kq*mq))
    return -np.sum(((kq*mq)/M)*np.log2((kq*mq)/M))/np.log2(M), -np.sum(((kq*mq)/M)*np.log2(kq/M))/np.log2(M)

# Define function needed for the Monte Carlo step which takes care of $k=0$
def dk_term_ldt(k, ldt_beta):
    k = float(k)
    if k!=0: return k*log(k) - loggamma(k+1) - ldt_beta*k*log(k)
    else: return - loggamma(k+1)

# Fix the parameters of the Dirichlet model
# $\rho = M/S_size$ where $S_size$ is the size of the state space and $M$ is the number of samples
rho = 10
M = rho*S_size
S_size = int(M/rho)

# Fix the large deviations parameter
ldt_beta = 1.0

HofK_sampling = np.zeros(100)
HofS_sampling = np.zeros(100)
DM_sample = np.zeros((100, len(k_sample)))
k_max = np.zeros(100)
iterations = 0
beta = 1.0 # sampling bias

# Sample 100 large deviations Dirichlet NML "typical" samples
while iterations < 100:
    # Initialize the vector $k_s$ which the frequency of the state $s$ in the sample $\hat{s}$
    # Here, we initialize by creating (almost) equally sampled states
    k_sample = np.floor(M/S_size)*np.ones(S_size)
    remainder = int(M - np.sum(k_sample))
    k_sample[0:remainder] = k_sample[0:remainder] + 1
    assert np.sum(k_sample)==M, 'sample problem'
    
    attempt = 0
    
    # Equilibrate to 2000*S_size before restarting the simulation
    while attempt < 2000*S_size:
        attempt += 1
    
        # propose to transfer one observation from k_1 to k_2
        k_1 = np.random.choice(np.where(k_sample>0)[0],1)
        k_2 = np.random.choice(np.arange(S_size),1)
    
        # Calculate the change in log-likelihood (of the tilted distribution) noting that only the states involved in the observation-transfer
        dk = (dk_term(k_sample[k_1]-1, ldt_beta) + dk_term(k_sample[k_2]+1, ldt_beta)) - (dk_term(k_sample[k_1], ldt_beta) + dk_term(k_sample[k_2], ldt_beta))
        if np.amin([exp(beta*dk),1.0]) >= float(np.random.rand()):
            k_sample[k_1] = k_sample[k_1] - 1
            k_sample[k_2] = k_sample[k_2] + 1
    
    # Sample large deviation and measure relevant quantities
    HofK, HofS = calculate_HofKS(k_sample)
    DM_sample[iterations] = k_sample
    k_max[iterations] = np.amax(k_sample)
    HofK_sampling[iterations] = HofK
    HofS_sampling[iterations] = HofS
    iterations += 1

DM_sample = np.array(DM_sample)
k_max = np.array(k_max)
HofK_sampling = np.array(HofK_sampling)
HofS_sampling = np.array(HofS_sampling)

