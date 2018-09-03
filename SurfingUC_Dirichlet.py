# Some basic imports
from __future__ import division

import numpy as np

from collections import Counter
from mpmath import mp, log, loggamma, exp, fmul, fsum, power

mp.dps = 100
mp.pretty = True

def calculate_HofKS(mapping_ks):
    ks_counts = np.asarray(Counter(mapping_ks).most_common())
    positive_values = np.where(ks_counts[:,0]>0)[0]
    kq, mq = ks_counts[:,0][positive_values], ks_counts[:,1][positive_values]
    assert np.sum(kq*mq)==np.sum(mapping_ks)
    M = float(np.sum(kq*mq))
    return -np.sum(((kq*mq)/M)*np.log2((kq*mq)/M))/np.log2(M), -np.sum(((kq*mq)/M)*np.log2(kq/M))/np.log2(M)

def dk_term(k):
    k = float(k)
    if k!=0: return k*log(k) - loggamma(k+1)
    else: return - loggamma(k+1)


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
