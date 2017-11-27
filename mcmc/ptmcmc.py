##### A demonstration of parallel tempered MCMC using the rosenbrock banana function in two dimensions

import random
import numpy as np
import scipy.stats as sci
import math as m
import matplotlib.pyplot as plt
import argparse
import triangle

pi = m.pi
exp = m.exp

parser = argparse.ArgumentParser(description='Set some MCMC Parameters')
mcmc = parser.add_argument_group('MCMC')

mcmc.add_argument('--iterations', default=1000,
	help='How many f*cking iterations?')
mcmc.add_argument('--step', default=0.03,
	help='Diagonal Covariance Entries for Proposal Distribution')
mcmc.add_argument('--nChains', default=3,
	help='Number of Chains for PTMCMC')
mcmc.add_argument('--temp', default=1.5,
	help='Temperature of First Tempered Chain')
mcmc.add_argument('--initPoint', default=None,
	help='Initial Guess Point for MCMC')
mcmc.add_argument('--swapRate', default=3,
	help='Inter-Chain Swap Proposal Rate')

args, unknown = parser.parse_known_args()

if args.iterations:
	N = int(args.iterations)
if args.step:
	step = float(args.step)
if args.nChains:
	nChains = int(args.nChains)
if args.initPoint:
	initPoint = tuple(args.initPoint)
if args.temp:
	temp0 = float(args.temp)
if args.swapRate:
	swp = int(args.swapRate)
else:
	temp0 = 1.5

def poop(x): # Testing a Gaussian Function
	gauss = sci.norm.pdf(x, loc=0.0, scale=1)
	return(gauss)

def nanner(point, temp): # Gaussian plus banana, small nChains and temp will miss banana mode
	mu, nu = point
	mu, nu = float(mu), float(nu)
        return ((16.0/(3*pi)*(exp(-(mu**2)-(9+4*mu**2+8*nu)**2)+0.5*exp(-(8*mu**2)-8*(nu-2)**2)))**(1/temp))

def step_MCMC(func, currPoint, temp, dim): # Produces one MCMC step, perhaps the original point, perhaps not
	covar = np.diag(step * np.ones(dim)) # Create a diagonal Covariance Matrix	
	propPoint = np.random.multivariate_normal(list(currPoint), covar) 
	draw = np.random.uniform()
	if draw <= min(1, (func(propPoint, temp) / func(currPoint, temp))):
		outPoint = propPoint
	else:
		outPoint = currPoint
	return outPoint 

def tempLadder(nChains): # Temperature spacing T_{i+1} = T_i ** c, as c goes from 1.2 -> 2
	tempExp = 0.8 / nChains
	temp = temp0
	temps = [1.0]
	for i in xrange(nChains - 1):
		temp = temp**(1.2 + tempExp)
		temps.append(temp)
	print('Using Temperature Ladder %r' % temps)
	return temps 

def PTMCMC(func, dim):
	totalCount = 0
	swapCount = 0
	ladder = tempLadder(nChains)
	currPoints = [initPoint for chains in range(nChains)]
	nextPoints = [(0,0) for x in xrange(nChains)]
	posterior = []
	for i in xrange(0, N - 1):
		for j in xrange(nChains):
			nextPoints[j] = step_MCMC(func,currPoints[j],ladder[j],dim)
		if (i % swp) == 0:

			totalCount += 1
			one = random.randint(0, nChains - 2)
			two = one + 1

			cool = func(currPoints[two], ladder[one]) / func(currPoints[one], ladder[one])
			warm = func(currPoints[two], ladder[two]) / func(currPoints[one], ladder[two])
			draw = np.random.uniform()

			if draw <= min(1, cool / warm):
				nextPoints[one], nextPoints[two] = nextPoints[two], nextPoints[one]
				swapCount += 1		
		currPoints = nextPoints	
		posterior.append(currPoints[0])
	print('Acceptance Ratio %r' % (float(swapCount) / float(totalCount)))
	import pdb
	pdb.set_trace()
	return posterior
			
posterior = (PTMCMC(nanner, 2))	
fig = triangle.corner(posterior, labels=[r'$\mu$', r'$\nu$'], plot_contours=True)
plt.show()





