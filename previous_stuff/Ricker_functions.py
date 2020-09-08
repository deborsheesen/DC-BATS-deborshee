from __future__ import division
get_ipython().magic(u'matplotlib inline')
import numpy as np, time, matplotlib.pyplot as plt, math, pandas, numpy.random as npr
from pylab import plot, show, legend
from scipy.stats import uniform
from time import time
from scipy.stats import *
from math import factorial
from tqdm import trange


def simulate_data(initial, T, theta):
    r, sigma, phi = theta[:]
    X = np.zeros(T+1); X[0] = initial 
    Y = np.zeros(T)
    for t in range(T):
        X[t+1] = X[t]*r*np.exp(-X[t] + sigma*npr.randn())
        Y[t] = npr.poisson(phi*X[t+1])
    return X[1::], Y.astype(int)


def propagate(particles, theta) :
    r, sigma, phi = theta[:]
    n_pf = len(particles)
    return particles*r*np.exp(-particles + sigma*npr.randn(n_pf))

def adaptive_resample(weights, particles) :
    weights /= np.sum(weights)
    ESS = 1/np.sum(weights**2)
    n_pf = len(weights)
    idx_resampled = np.arange(n_pf)
    if ESS < n_pf/2 :
        particles = particles[npr.choice(a=n_pf,size=n_pf,p=weights)]
        weights = np.ones(n_pf)/n_pf
    return weights, particles 

def potential(particles, y, theta) :
    r, sigma, phi = theta[:]
    return np.exp(-phi*particles)*(phi*particles)**y/float(factorial(y))
    

def bootstrap_PF_simple(initial, n_pf, theta, Y) :
    T = len(Y)
    r, sigma, phi = theta[:]
    particles, weights, logNC = np.zeros(n_pf), np.ones(n_pf), 0.
    particles[:] = initial 
    
    for t in range(T) :
        particles = propagate(particles, theta)
        incremental_weights = potential(particles, Y[t], theta)
        weights = weights*incremental_weights
        logNC += np.log(np.sum(weights))
        weights, particles = adaptive_resample(weights, particles)
                
    return logNC


def bootstrap_PF(initial, n_pf, theta, Y) :
    T = len(Y)
    r, sigma, phi = theta[:]
    particles, weights, logNC = np.zeros((T+1,n_pf)), np.ones(n_pf), 0.
    particles[0] = initial 
    
    for t in range(T) :
        particles[t+1] = propagate(particles[t], theta)
        incremental_weights = potential(particles[t+1], Y[t], theta)
        weights = weights*incremental_weights
        logNC += np.log(np.sum(weights))
        weights, particles[t+1] = adaptive_resample(weights, particles[t+1])
                
    return logNC, particles, weights

def log_prior(theta) :
    return 0 



def pMCMC(initial, Y, theta_0, n_pf, n_mcmc, scale, power=1, adapt=True, start_adapt=0.2) :
    
    theta_dim = len(theta_0);
    theta_chain = np.zeros((n_mcmc+1,theta_dim))
    theta_chain[0] = theta_0
    log_theta_mu, log_theta_m2 = np.log(theta_0), np.log(theta_0)**2
    lls = np.zeros(n_mcmc+1) 
    lls[0] = bootstrap_PF_simple(initial, n_pf, theta_chain[0], Y)
    scales = np.ones((n_mcmc+1,theta_dim))
    scales[:] = scale
    accepted = 0
    
    last_jump = 0
    
    for n in trange(n_mcmc) :
        
        theta_proposed = np.exp(np.log(theta_chain[n]) + scales[n]*npr.randn(theta_dim)) 
        ll_proposed = bootstrap_PF_simple(initial, n_pf, theta_proposed, Y)
        log_prior_current, log_prior_proposed = log_prior(theta_chain[n]), log_prior(theta_proposed) 
        log_accept_prob = power*(ll_proposed-lls[n]) + (log_prior_proposed-log_prior_current) + np.log(np.prod(theta_proposed/theta_chain[n]))
        
        if np.log(npr.rand()) < log_accept_prob :
            lls[n+1], theta_chain[n+1] = ll_proposed, theta_proposed
            accepted += 1
            latest_jump = n
        else :
            lls[n+1], theta_chain[n+1] = lls[n], theta_chain[n]
            if n - last_jump > 100 :
                lls[n+1] = bootstrap_PF_simple(initial, n_pf, theta_chain[n+1], Y)
                
        log_theta_mu = ((n+1)*log_theta_mu + np.log(theta_chain[n+1]))/(n+2)
        log_theta_m2 = ((n+1)*log_theta_m2 + np.log(theta_chain[n+1])**2)/(n+2)
        if adapt :
            if n >= int(n_mcmc*start_adapt) : 
                scales[n+1] = np.sqrt((log_theta_m2 - log_theta_mu**2))*0.7

    print(100*accepted/n_mcmc, "% acceptance rate")
    return theta_chain, scales


def chunked_pMCMC(x_0, Y, theta_0, n_mcmc, scale, n_pf, chunk_size, power=1, N_init=1000, adapt=True, start_adapt=0.2) :
    
    T = len(Y)
    n_chunks = int(T/chunk_size)
    theta_chains = np.zeros((n_chunks, n_mcmc+1, len(theta_0)))
    
    #Run pseudo-margial MCMC on first chunk:
    theta_chain = pMCMC(x_0, Y[:chunk_size], theta_0, n_pf, n_mcmc, scale, power, adapt, start_adapt)[0]
    theta_chains[0] = theta_chain
    
    # Iterate over remaining chunks:
    for i in np.arange(1,n_chunks) :
        
        # Generate starting points for particle filter
        result = bootstrap_PF(np.exp(5*npr.randn(N_init)), N_init, np.mean(theta_chain, 0), Y[i*chunk_size-5:i*chunk_size]) 

        XX, w = result[1], result[2]
        initial = XX[-1, npr.choice(N_init,n_pf,p=w/sum(w))]
        
        # Run pseudo-marginal MCMC with these starting points on chunks
        theta_chain_apprx = pMCMC(initial, Y[i*chunk_size:(i+1)*chunk_size], theta_0, n_pf, n_mcmc, scale, power, adapt, start_adapt)[0]
        theta_chains[i] = theta_chain_apprx
    
    return theta_chains 

def chunked_pMCMC_true(Y, X, theta_0, n_mcmc, scale, n_pf, chunk_size, power=1, N_init=10_000, init_sigma=3, adapt=True) :
    
    T = len(Y)
    n_chunks = int(T/chunk_size)
    theta_chains = np.zeros((n_chunks, n_mcmc+1, 3))
    
    theta_chains[0] = pMCMC(X[0], Y[:chunk_size], theta_0, n_pf, n_mcmc, scale, power, adapt)[0]
    
    # Iterate over chunks
    for i in np.arange(1,n_chunks) :
        # Run pseudo-marginal MCMC with these starting points on chunks
        theta_chains[i] = pMCMC(X[i*chunk_size-1], Y[i*chunk_size:(i+1)*chunk_size], theta_0, n_pf, n_mcmc, scale, power, adapt)[0]
    
    return theta_chains 







