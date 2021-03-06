# UniversalCodes

These Python codes were used to generate the typical and atypical samples from the normalized maximum likelihood (NML, in the sense of the Minimum Description Length principle) of different statistical models: Dirichlet model, independent spin model, Sherrington-Kirkpatrick model and restricted Boltzmann machine.

The details of the calculations are found in
> R.J. Cubero, M. Marsili and Y. Roudi <br>
> Minimum Description Length codes are critical <br>
> https://arxiv.org/abs/1809.00652 <br>

This repository contains the following files needed to reproduce the results of the paper: <br>
-**SurfingUC_Dirichlet.py** generates samples drawn from the NML distribution of the Dirichlet model using a Markov chain Monte Carlo (MCMC) approach. One needs to define the parameters rho (=M/S_size, average number of points per state), S_size (the size of the state space) and M (the size of the sample). <br> <br>
-**SurfingUC_Paramagnet.py** generates samples drawn from the NML distribution of the independent spin model (paramagnet) using the MCMC approach. Because of the independence between spins, we separately generate the configurations of each spin constrained by the number of up-spins drawn from the NML distribution. Here, one has to define the parameter M (the size of the sample) and the range of the number of spins, n, in the system. <br> <br>
-**SurfingUC_SK.py** generates samples drawn from the NML distribution of the Sherrington-Kirkpatrick model (pairwise spin model with variable interaction strength and local field). <br> <br>
-**SurfingUC_rbm.py** generates samples drawn from the NML distribution of the restricted Boltzmann machine (a two-layered network architechture where one has n_v independent variables interacting in the visible layer with n_h independent variables in the hidden layer). <br> <br>
-**LD_Dirichlet.py** generates large deviation samples from the NML distribution of the Dirichlet model using an MCMC approach. Here, one can play with the parameter beta which explores the large deviation events from the typical H[s] of the Dirichlet NML codes. <br> <br> <br>

-**LD_SK.py** generates large deviation samples from the NML distribution of the Sherrington-Kirkpatrick model using an MCMC approach. Note that the results of this simulation are supplementary to the paper. <br> <br>
-**LD_rbm.py** generates large deviation samples from the NML distribution of the restricted Boltzmann machine using an MCMC approach. Note that the results of this simulation are supplementary to the paper.
