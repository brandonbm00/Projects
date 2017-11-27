import numpy as np
import scipy.stats as sci
import matplotlib.pyplot as plt
import math as m
from sys import argv
import argparse
import triangle


parser = argparse.ArgumentParser(description='Set Some MCMC Tester Parameters')
mcmc = parser.add_argument_group('MCMC')

mcmc.add_argument('--testmu', default=0,
	help='What MU would you like to use?')
mcmc.add_argument('--testsig', default=1,
	help='What SIGMA would you like to use?')
mcmc.add_argument('--iterations', default=10000,
	help='How many ITERATIONS would you like?')
mcmc.add_argument('--step', default=1, 
	help='What stepsize would you like?')

args, unknown = parser.parse_known_args()

pi = m.pi

if args.iterations:
	N = int(args.iterations)
if args.testmu:
	test_mu = float(args.testmu)
if args.testsig:
	test_sig = float(args.testsig)
if args.step:
	step = float(args.step)


##### MCMC to sample from a gaussian function
sci_norm_pdf = sci.norm.pdf

def stdNormal(x,mu,sig):
	return sci_norm_pdf(x, loc=mu, scale=sig)

def gaussianLikelihood(xlist, mu, sig):
	norms = stdNormal(xlist, mu, sig)
	lognorms = np.log(norms)
	L = np.sum(lognorms) 
	if np.isnan(L):
		L = -1000000000.0
	return L

def prior(mu, sig):
	priorMu = 1.0
	if 0 <= sig <= 20:
		priorSig = 1.0
	else:
		priorSig = 0.0
	return(priorMu*priorSig)

##### Metropolis-Hastings Routine for sampling from a standard gaussian - later use this data to infer original parameters


x = np.zeros(N)
x[0] = 0 

for i in xrange(0, N - 1):

	x_proposed = np.random.normal(x[i], step)
	draw = np.random.uniform()

	if draw <= min(1, sci.norm.pdf(x_proposed, loc=test_mu, scale=test_sig) / sci.norm.pdf(x[i], loc=test_mu, scale=test_sig)):
		x[i + 1] = x_proposed
	else:
		x[i + 1] = x[i]

plt.hist(x, bins=50)
plt.show()

data = x

##### Now time to infer on dem params	

initial_guess = [0, 1]


##### Initialization of mu and sigma lists
mu_list = np.zeros(N)
sig_list = np.zeros(N)
log_likelihoods = np.zeros(N)

mu_list[0] = (initial_guess[0])
sig_list[0] = (initial_guess[1])
log_likelihoods = []

##### Main Lewp

covar = np.array([[0.03, 0], [0, 0.03]])

def MCMC():
	for i in xrange(0, N - 1):

		muProposed, sigProposed = np.random.multivariate_normal([mu_list[i], sig_list[i]], covar)
		draw = np.random.uniform()

		if np.log(draw) <= min(0.0, (gaussianLikelihood(data, muProposed, sigProposed) - gaussianLikelihood(data, mu_list[i], sig_list[i]))):
			mu_list[i + 1] = muProposed
			sig_list[i + 1] = sigProposed
			log_likelihoods.append(gaussianLikelihood(data, mu_list[i+1], sig_list[i+1]))		
		else:
			mu_list[i + 1] = mu_list[i]
			sig_list[i + 1] = sig_list[i]
	return(mu_list, sig_list)

mu_list, sig_list = MCMC()
plt.scatter(mu_list, sig_list)
plt.show()

plt.plot(np.arange(len(log_likelihoods)), log_likelihoods)
plt.show()








	
