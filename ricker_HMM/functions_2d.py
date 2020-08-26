
# coding: utf-8

# In[1]:

# Calling libraries:
from __future__ import division
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.cm as cm
import scipy
from scipy.sparse.linalg import inv
import scipy.spatial as spatial
from scipy.stats import *

def mutate(particles, delta, dt, theta) :
    alpha, sigma, sigma_error = theta[:]
    N = np.shape(particles)[0]
    X = np.copy(particles)
    sqdt = np.sqrt(dt)
    for i in range(int(delta/dt)) :
        W = np.random.randn(N,2)  
        r = np.sqrt(np.sum(X**2,axis=1))
        X[:,0] += -alpha*X[:,0]*dt + sqdt*( np.sin(sigma*r)*W[:,0] - np.cos(sigma*r)*W[:,1] )
        X[:,1] += -alpha*X[:,1]*dt + sqdt*( np.cos(sigma*r)*W[:,0] + np.sin(sigma*r)*W[:,1] )
    return X

def simulate_data(theta, x_0, delta, dt, T) :
    
    alpha, sigma, sigma_error = theta
    hidden = np.zeros((T+1, len(x_0)))
    hidden[0] = x_0
    observed = np.zeros(T)
    
    for t in range(T) :
        hidden[t+1] = mutate(hidden[t].reshape(1,2), delta, dt, theta)        
        observed[t] = hidden[t+1,1] + sigma_error*np.random.randn(1)
    
    return hidden, observed 

def bootstrap_PF(delta, dt, g, y, theta, x_0, N) :
    
    T = np.shape(y)[0]
    
    alpha, sigma, sigma_error = theta
    particles = np.zeros((T+1, N, 2)) 
    particles[0] = x_0
    weights   = np.ones(N)/N
    log_NC    = np.zeros(T+1) 
    
    for t in range(T):        
        # mutate:
        particles[t+1] = mutate(particles[t], delta, dt, theta)
        weights   *= g(y[t], particles[t+1], sigma_error)
        log_NC[t+1] = log_NC[t] + np.log(np.sum(weights))
        weights   /= np.sum(weights)       
        
        # resample:
        if 1/np.sum(weights**2) < N/2 : 
            particles[t+1,:] = particles[t+1, np.random.choice(N,N,True,weights)] 
            weights = np.ones(N)/N
        
    return particles, weights, log_NC

def g(y, X, sigma_error):
    return norm.pdf(x=y, loc=X[:,1], scale=sigma_error)

def prior(theta, prior_mean, prior_sd) :
    log_theta = np.log(theta)
    log_mean = np.log(prior_mean)
    return np.prod(norm.pdf(log_theta, log_mean, prior_sd)/theta)
    
def pseudo_marginal_MCMC(prior_mean, prior_sd, delta, dt, y, scale, theta_0, x_0, n_particles, n_mcmc) :
    
    indices_sd, indices_scale = prior_sd>0, scale>0
    assert (indices_sd == indices_scale).sum() == 3
    
    theta_chain = np.zeros((n_mcmc+1, 3)); theta_chain[0] = theta_0[:]
    lls = np.zeros(n_mcmc+1) 
    lls[0] = bootstrap_PF(delta, dt, g, y, theta_chain[0], x_0, n_particles)[-1][-1]
    
    for n in range(n_mcmc) :
        theta_proposed = np.exp( np.log(theta_chain[n]) + scale*np.random.randn(3) ) 
        ll_proposed = bootstrap_PF(delta, dt, g, y, theta_proposed, x_0, n_particles)[-1][-1] 
        alpha = (np.exp(ll_proposed-lls[n])*(np.prod(theta_proposed/theta_chain[n]))*
                 prior(theta_proposed[indices_sd], prior_mean[indices_sd], prior_sd[indices_sd])/prior(theta_chain[n,indices_sd], prior_mean[indices_sd], prior_sd[indices_sd]))
        if uniform.rvs(0,1,1) < alpha :
            lls[n+1], theta_chain[n+1] = ll_proposed, theta_proposed[:]
        else :
            lls[n+1], theta_chain[n+1] = lls[n], theta_chain[n]
    return theta_chain

def representative_jumps(delta, dt, g, theta, scale, y, x_0, n_particles, rep, n_jumps) :
    lls = np.zeros((rep, n_jumps))
    
    for n in range(n_jumps) :
        ll_0 = bootstrap_PF(delta, dt, g, y, theta, x_0, n_particles)[-1][-1]
        theta_proposed = theta*np.exp(scale*np.random.randn(len(theta)))
        for r in range(rep) :
            lls[r,n] = bootstrap_PF(delta, dt, g, y, theta_proposed, x_0, n_particles)[-1][-1] - ll_0
    
    return lls.var(axis=0).mean()