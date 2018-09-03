# Some basic imports
from __future__ import division

import numpy as np

from collections import Counter
from mpmath import mp, log, loggamma, psi, exp, fmul, fsum, power, sqrt, pi

mp.dps = 100
mp.pretty = True

def pofk_paramagnet(N, k):
    preterm = sqrt(2./(pi*N))
    if k==0:
        exp_term = loggamma(N+1) - loggamma(k+1) - loggamma(N-k+1)
    elif k==N:
        exp_term = loggamma(N+1) - loggamma(k+1) - loggamma(N-k+1)
    else:
        exp_term = loggamma(N+1) - loggamma(k+1) - loggamma(N-k+1) + k*log(k/N) + (N-k)*log(1.-(k/N))
    return preterm*exp(exp_term)


rho = 10
S_size = 1000
M = rho*S_size

k_sample = np.floor(M/S_size)*np.ones(S_size)
remainder = int(M - np.sum(k_sample))
k_sample[0:remainder] = k_sample[0:remainder] + 1
assert np.sum(k_sample)==M, 'sample problem'

HofK_sampling = []
HofS_sampling = []
accept = 0
attempt = 0

beta = 1.0

while attempt<1000000:
    attempt = attempt + 1
    k_1 = np.random.choice(np.where(k_sample>0)[0],1)
    k_2 = np.random.choice(np.arange(S_size),1)
    dk = (dk_term(k_sample[k_1]-1) + dk_term(k_sample[k_2]+1)) - (dk_term(k_sample[k_1]) + dk_term(k_sample[k_2]))
    if np.amin([exp(beta*dk),1.0]) >= float(np.random.rand()):
        accept = accept+1
        k_sample[k_1] = k_sample[k_1] - 1
        k_sample[k_2] = k_sample[k_2] + 1
        accept += 1
    
    if ((attempt >= 0) and (np.mod(attempt,100)==0)):
        HofK, HofS = calculate_HofKS(k_sample)
        HofK_sampling.append(HofK)
        HofS_sampling.append(HofS)
