from __future__ import division

import numpy as np
import operator, functools

# redefine spin states into integers
def s2i_states(data):
    # transform data from {-1,+1} to {0, 1}
    binary_data = (data+1)/2
    
    # binary state to integer state
    binary_state = binary_data.dot(1 << np.arange(binary_data.shape[-1] - 1, -1, -1))
    binary_state = binary_state.astype('int')

    return binary_state

def b2i_states(data):
    return data.dot(1 << np.arange(data.shape[-1] - 1, -1, -1)).astype('int')

# define bit test function
def btest(i, pos):
    return bool(i & (1 << pos))

# defines the operators
def phi(mu, s, N):
    return 1-2*np.mod(np.sum([int(btest(mu,i) and not btest(s,i)) for i in np.arange(N)]),2)

# define integer to binary function
def binary(mu, N):
    return ("{0:0"+str(N)+"b}").format(mu)

def binary_array(mu, N):
    return np.array(list(binary(mu,N))).astype("int")

# define spin flipping
def spin_reversal(zipped_data):
    s, N, pos = zipped_data
    spin_str = binary(s,N)
    return int(spin_str[:pos] + str(np.mod(int(spin_str[pos])+1,2)) + spin_str[pos+1:],2)
