# ------------------------------------------------------------------- #
# This Python code samples the sample space $\chi$ of the normalized  #
# maximum likelihood (NML) for the independent spin model which is    #
# used to plot Figure 3 in the text:                                  #
#    Minimum Description Length codes are critical                    #
#    Cubero, RJ; Marsili, M; Roudi, Y                                 #
# ------------------------------------------------------------------- #

# Some basic imports
from __future__ import division
import numpy as np

# Import necessary libraries
from collections import Counter
from mpmath import mp, log, loggamma, psi, exp, fmul, fsum, power, sqrt, pi

# Configure mpmath to calculate quantities up to 100 decimal places
mp.dps = 100
mp.pretty = True

# Define function needed for the Monte Carlo step which takes care of $k=0$ and $k=N$
def pofk_paramagnet(N, k):
    preterm = sqrt(2./(pi*N))
    if k==0:
        exp_term = loggamma(N+1) - loggamma(k+1) - loggamma(N-k+1)
    elif k==N:
        exp_term = loggamma(N+1) - loggamma(k+1) - loggamma(N-k+1)
    else:
        exp_term = loggamma(N+1) - loggamma(k+1) - loggamma(N-k+1) + k*log(k/N) + (N-k)*log(1.-(k/N))
    return preterm*exp(exp_term)

# Define the size $M$ of the sample $\hat{s}$
M = 10000

# Create the NML code for the paramagnet for a single spin
k_range = np.arange(0,M+1,1).astype('float')
pofk = [pofk_paramagnet(M, k) for k in k_range]
norm_pofk = np.sum(pofk)
pofk = [p/norm_pofk for p in pofk]

HofK_paramagnet = {}
HofS_paramagnet = {}
samples = {}

# Iterate over a range of number of spins, $n$, in the system
for n_spins in np.arange(3,21,1):
    HofS_paramagnet[n_spins] = []
    HofK_paramagnet[n_spins] = []
    samples[n_spins] = []
    for iterations in np.arange(100):
        # Draw the number of up-spins from the NML code of the paramagnet
        k_plus = np.random.choice(k_range, p=pofk, size=n_spins).astype('int')
        
        # Create the spin configuration of the system
        # Note that the only important quantity here is the number of up-spins
        data = np.zeros((n_spins, M))
        for i in np.arange(n_spins):
            spins = np.zeros(M)
            spins[0:k_plus[i]] = 1
            data[i] = np.random.permutation(spins)
        
        # Calculate the relevant quantities
        ks = [v for u,v in Counter(b2i_states(data.T)).items()]
        HofK, HofS = calculate_HofKS(ks)
        HofS_paramagnet[n_spins].append(HofS)
        HofK_paramagnet[n_spins].append(HofK)
        samples[n_spins].append(b2i_states(data.T))
