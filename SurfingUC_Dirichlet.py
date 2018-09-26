# ------------------------------------------------------------------- #
# This Python code samples the sample space $\chi$ of the normalized  #
# maximum likelihood (NML) for the Dirichlet model which is used to   #
# plot Figure 2 in the text:                                          #
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
def dk_term(k):
    k = float(k)
    if k!=0: return k*log(k) - loggamma(k+1)
    else: return - loggamma(k+1)

# Fix the parameters of the Dirichlet model
# $\rho = M/S_size$ where $S_size$ is the size of the state space and $M$ is the number of samples
rho = 10
S_size = 1000
M = rho*S_size

output_name = 'dirichlet_M1e3_S1000'

# Initialize the vector $k_s$ which the frequency of the state $s$ in the sample $\hat{s}$
# Here, we initialize by creating (almost) equally sampled states
k_sample = np.floor(M/S_size)*np.ones(S_size)
remainder = int(M - np.sum(k_sample))
k_sample[0:remainder] = k_sample[0:remainder] + 1
assert np.sum(k_sample)==M, 'sample problem'


HofK_sampling = np.zeros(100)
HofS_sampling = np.zeros(100)
DM_sample = np.zeros((100, len(k_sample)))
accept = 0
attempt = 0
iterations = 0
beta = 1.0 # sampling bias

# Sample 100 Dirichlet NML typical samples
while iterations < 100:
    attempt = attempt + 1
    
    # propose to transfer one observation from k_1 to k_2
    k_1 = np.random.choice(np.where(k_sample>0)[0],1)
    k_2 = np.random.choice(np.arange(S_size),1)
    
    # Calculate the change in log-likelihood noting that only the states involved in the observation-transfer
    dk = (dk_term(k_sample[k_1]-1) + dk_term(k_sample[k_2]+1)) - (dk_term(k_sample[k_1]) + dk_term(k_sample[k_2]))
    
    # Do the Monte Carlo step
    if np.amin([exp(beta*dk),1.0]) >= float(np.random.rand()):
        k_sample[k_1] = k_sample[k_1] - 1
        k_sample[k_2] = k_sample[k_2] + 1
        accept += 1 # this is to follow the acceptance ratio
    
    # Equilibrate the Markov chain Monte Carlo at 5000*S_size before calculating relevant quantities
    if ((attempt >= 5000*S_size) and (np.mod(attempt,50*S_size)==0)):
        HofK, HofS = calculate_HofKS(k_sample)
        DM_sample[iterations] = k_sample
        HofK_sampling[iterations] = HofK
        HofS_sampling[iterations] = HofS
        iterations += 1

DM_samples = np.array(DM_samples)
HofK_sampling = np.array(HofK_sampling)
HofS_sampling = np.array(HofS_sampling)

np.savetxt('HofS_'+output_name+'.d', HofS_sampling)
np.savetxt('HofK_'+output_name+'.d', HofK_sampling)
np.savetxt('UC_'+output_name+'.d', DM_samples)
