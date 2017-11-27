import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
import math as m

def stdNormal(x, mu, sig):
	return sci.norm.pdf(x, loc=mu, scale=sig)

def gaussianLikelihood(xlist, mu, sig):
	L = np.sum(np.log(stdNormal(xlist, mu, sig)))	
	if np.isnan(L):
		L = -10000000000.0
	return L

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

