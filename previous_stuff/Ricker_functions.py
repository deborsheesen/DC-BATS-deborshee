
# coding: utf-8

# In[1]:


# Calling libraries:
from __future__ import division
get_ipython().magic(u'matplotlib inline')
import numpy as np, time, matplotlib.pyplot as plt, math, pandas, numpy.random as npr
from pylab import plot, show, legend
from scipy.stats import uniform
from time import time
from scipy.stats import *
from math import factorial
from tqdm import trange


def simulate_data(initial, T, θ):
    r, σ, ϕ = θ[:]
    X = np.zeros(T+1); X[0] = initial 
    Y = np.zeros(T)
    for t in range(T):
        X[t+1] = X[t]*r*np.exp(-X[t] + σ*npr.randn())
        Y[t] = npr.poisson(ϕ*X[t+1])
    return X[1::], Y.astype(int)


def propagate(particles, θ) :
    r, σ, ϕ = θ[:]
    n_pf = len(particles)
    return particles*r*np.exp(-particles + σ*npr.randn(n_pf))

def adaptive_resample(weights, particles) :
    weights /= np.sum(weights)
    ESS = 1/np.sum(weights**2)
    n_pf = len(weights)
    idx_resampled = np.arange(n_pf)
    if ESS < n_pf/2 :
        particles = particles[npr.choice(a=n_pf,size=n_pf,p=weights)]
        weights = np.ones(n_pf)/n_pf
    return weights, particles 

def potential(particles, y, θ) :
    r, σ, ϕ = θ[:]
    return np.exp(-ϕ*particles)*(ϕ*particles)**y/float(factorial(y))
    

def bootstrap_PF_simple(initial, n_pf, θ, Y) :
    T = len(Y)
    r, σ, ϕ = θ[:]
    particles, weights, logNC = np.zeros(n_pf), np.ones(n_pf), 0.
    particles[:] = initial 
    
    for t in range(T) :
        particles = propagate(particles, θ)
        incremental_weights = potential(particles, Y[t], θ)
        weights = weights*incremental_weights
        logNC += np.log(np.sum(weights))
        weights, particles = adaptive_resample(weights, particles)
                
    return logNC


def bootstrap_PF(initial, n_pf, θ, Y) :
    T = len(Y)
    r, σ, ϕ = θ[:]
    particles, weights, logNC = np.zeros((T+1,n_pf)), np.ones(n_pf), 0.
    particles[0] = initial 
    
    for t in range(T) :
        particles[t+1] = propagate(particles[t], θ)
        incremental_weights = potential(particles[t+1], Y[t], θ)
        weights = weights*incremental_weights
        logNC += np.log(np.sum(weights))
        weights, particles[t+1] = adaptive_resample(weights, particles[t+1])
                
    return logNC, particles, weights

def log_prior(θ) :
    return 0 



def pMCMC(initial, Y, θ_0, n_pf, n_mcmc, scale, power=1, adapt=True, start_adapt=2000) :
    
    theta_dim = len(θ_0);
    θ_chain = np.zeros((n_mcmc+1,theta_dim))
    θ_chain[0] = θ_0
    log_θ_mu, log_θ_m2 = np.log(θ_0), np.log(θ_0)**2
    lls = np.zeros(n_mcmc+1) 
    lls[0] = bootstrap_PF_simple(initial, n_pf, θ_chain[0], Y)
    scales = np.ones((n_mcmc+1,theta_dim))
    scales[:start_adapt] = scale
    
    for n in trange(n_mcmc) :
        
        θ_proposed = np.exp(np.log(θ_chain[n]) + scales[n]*npr.randn(theta_dim)) 
        ll_proposed = bootstrap_PF_simple(initial, n_pf, θ_proposed, Y)
        log_prior_current, log_prior_proposed = log_prior(θ_chain[n]), log_prior(θ_proposed) 
        log_accept_prob = power*(ll_proposed-lls[n]) + (log_prior_proposed-log_prior_current) + np.log(np.prod(θ_proposed/θ_chain[n]))
        
        if np.log(npr.rand()) < log_accept_prob :
            lls[n+1], θ_chain[n+1] = ll_proposed, θ_proposed
        else :
            lls[n+1], θ_chain[n+1] = lls[n], θ_chain[n]
        log_θ_mu = ((n+1)*log_θ_mu + np.log(θ_chain[n+1]))/(n+2)
        log_θ_m2 = ((n+1)*log_θ_m2 + np.log(θ_chain[n+1])**2)/(n+2)
        if adapt :
            #if n == 200 : print(log_θ_m2, log_θ_mu**2, np.sqrt((log_θ_m2 - log_θ_mu**2)))
            if n >= start_adapt : 
                scales[n+1] = np.sqrt((log_θ_m2 - log_θ_mu**2))*0.7

    return θ_chain, scales


def chunked_pMCMC(initial, Y, θ_0, n_mcmc, scale, n_pf, chunk_size, power=1, N_init=10_000, init_σ=3, adapt=True) :
    
    T = len(Y)
    n_chunks = int(T/chunk_size)
    θ_chains = np.zeros((n_chunks, n_mcmc+1, len(θ_0)))
    
    
    #Run pseudo-margial MCMC on first chunk:
    θ_chain = pMCMC(initial, Y[:chunk_size], θ_0, n_pf, n_mcmc, scale, power, adapt)[0]
    θ_chains[0] = θ_chain
    
    # Iterate over remaining chunks:
    for i in np.arange(1,n_chunks) :
        
        # Generate starting points for particle filter
        result = bootstrap_PF(np.exp(init_σ*npr.randn(N_init)), N_init, 
                              np.mean(θ_chain, axis=0), Y[i*chunk_size-5:i*chunk_size]) 

        XX, w = result[1], result[2]
        initial = XX[-1, npr.choice(N_init,n_pf,p=w/sum(w))]
        
        # Run pseudo-marginal MCMC with these starting points on chunks
        θ_chain_apprx = pMCMC(initial, Y[i*chunk_size:(i+1)*chunk_size], θ_0, n_pf, n_mcmc, scale, power, adapt)[0]
        θ_chains[i] = θ_chain_apprx
    
    return θ_chains 

def chunked_pMCMC_true(Y, X, θ_0, n_mcmc, scale, n_pf, chunk_size, power=1, N_init=10_000, init_σ=3, adapt=True) :
    
    T = len(Y)
    n_chunks = int(T/chunk_size)
    θ_chains = np.zeros((n_chunks, n_mcmc+1, 3))
    
    θ_chains[0] = pMCMC(X[0], Y[:chunk_size], θ_0, n_pf, n_mcmc, scale, power, adapt)[0]
    
    # Iterate over chunks
    for i in np.arange(1,n_chunks) :
        # Run pseudo-marginal MCMC with these starting points on chunks
        θ_chains[i] = pMCMC(X[i*chunk_size-1], Y[i*chunk_size:(i+1)*chunk_size], θ_0, n_pf, n_mcmc, scale, power, adapt)[0]
    
    return θ_chains 







