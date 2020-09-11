from __future__ import division
import numpy as np, time, matplotlib.pyplot as plt, math, pandas, numpy.random as npr, multiprocessing as mp
from time import time
from scipy.stats import *
from tqdm import trange


def potential(y, particles, theta) :
    alpha, lmbda = theta[0], theta[1]
    reg = (alpha + lmbda.dot(particles)).transpose()
    return np.sum(np.exp((y-1)*reg), (1,2))

def propagate(particles, theta) :
    c, phi, logsigmasq = theta[2], theta[3], theta[4]
    sigmasq = np.exp(logsigmasq)
    return c + phi*particles + np.sqrt(sigmasq)*npr.randn(*np.shape(particles)) 

def simulate_data(x_0, T, n_species, theta) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    
    n_locations, n_factors = np.shape(x_0)
    Y = np.zeros((T,n_locations,n_species))
    X = np.zeros((T+1,n_locations,n_factors))
    X[0] = x_0

    for t in range(T) :
        X[t+1] = propagate(X[t], theta)
        reg = alpha + lmbda.dot(X[t].transpose())
        probs = 1/(1+np.exp(-reg))
        Y[t] = npr.binomial(n=1, p=probs).transpose()
    
    return Y, X

def adaptive_resample(weights, particles) :
    weights /= np.sum(weights)
    ESS = 1/np.sum(weights**2)
    n_particles = len(weights)
    idx_resampled = np.arange(n_particles)
    if ESS < n_particles/2 :
        particles = particles[:,:,npr.choice(a=n_particles,size=n_particles,p=weights)]
        weights = np.ones(n_particles)/n_particles
    return weights, particles 

def bootstrap_PF(x_0, n_particles, theta, Y) :
    T, n_locations, n_species = np.shape(Y)
    particles, weights, logNC = np.zeros((*np.shape(x_0),n_particles)), np.ones(n_particles), 0.
    for n in range(n_particles) :
        particles[:,:,n] = x_0
    for t in range(T) :
        particles = propagate(particles, theta)
        incremental_weights = potential(Y[t], particles, theta)
        weights = weights*incremental_weights
        logNC += np.log(np.sum(weights))
        weights, particles = adaptive_resample(weights, particles)                
    return logNC

def initialise(theta_0, n_mcmc) :
    alpha, lmbda, c, phi, logsigmasq = theta_0[:]
    
    alpha_chain = np.zeros(n_mcmc+1)
    lmbda_chain = np.zeros((n_mcmc+1,*np.shape(lmbda)))
    c_chain = np.zeros(n_mcmc+1)
    phi_chain = np.zeros(n_mcmc+1)
    logsigmasq_chain = np.zeros(n_mcmc+1)
    
    alpha_chain[0] = alpha
    lmbda_chain[0] = lmbda
    c_chain[0] = c
    phi_chain[0] = phi
    logsigmasq_chain[0] = logsigmasq
    
    lls = np.zeros(n_mcmc+1) 
    
    theta_mu = np.copy(theta_0)
    theta_m2 = np.array([alpha**2, lmbda**2, c**2, phi**2, logsigmasq**2], dtype=object)

    return alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain, lls, theta_mu, theta_m2

def propose(theta, scale) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    scale_alpha, scale_lmbda, scale_c, scale_phi, scale_logsigmasq = scale[:]
    
    alpha_proposed = alpha + scale_alpha*npr.randn()
    lmbda_proposed = lmbda + scale_lmbda*npr.randn(*np.shape(lmbda))
    c_proposed = c + scale_c*npr.randn()
    phi_proposed = phi + scale_phi*npr.randn()
    logsigmasq_proposed = logsigmasq + scale_logsigmasq*npr.randn()
    
    return [alpha_proposed, lmbda_proposed, c_proposed, phi_proposed, logsigmasq_proposed]

def update_moments(theta_mu, theta_m2, theta_new, n) :
    alpha_mu, lmbda_mu, c_mu, phi_mu, logsigmasq_mu = theta_mu
    alpha_m2, lmbda_m2, c_m2, phi_m2, logsigmasq_m2 = theta_m2
    alpha_new, lmbda_new, c_new, phi_new, logsigmasq_new = theta_new[:]
    
    alpha_mu = (n*alpha_mu + alpha_new)/(n+1)
    lmbda_mu = (n*lmbda_mu + lmbda_new)/(n+1)
    c_mu = (n*c_mu + c_new)/(n+1)
    phi_mu = (n*phi_mu + phi_new)/(n+1)
    logsigmasq_mu = (n*logsigmasq_mu + logsigmasq_new)/(n+1)
    
    alpha_m2 = (n*alpha_m2 + alpha_new**2)/(n+1)
    lmbda_m2 = (n*lmbda_m2 + lmbda_new**2)/(n+1)
    c_m2 = (n*c_m2 + c_new**2)/(n+1)
    phi_m2 = (n*phi_m2 + phi_new**2)/(n+1)
    logsigmasq_m2 = (n*logsigmasq_m2 + logsigmasq_new**2)/(n+1)
    
    theta_mu = [alpha_mu, lmbda_mu, c_mu, phi_mu, logsigmasq_mu]
    theta_m2 = [alpha_m2, lmbda_m2, c_m2, phi_m2, logsigmasq_m2]
    
    return theta_mu, theta_m2

def adapt_scale(scale, theta_mu, theta_m2) :
    scale_alpha, scale_lmbda, scale_c, scale_phi, scale_logsigmasq = scale[:]
    alpha_mu, lmbda_mu, c_mu, phi_mu, logsigmasq_mu = theta_mu
    alpha_m2, lmbda_m2, c_m2, phi_m2, logsigmasq_m2 = theta_m2
    
    scale_alpha = np.sqrt(alpha_m2 - alpha_mu**2)*0.7
    scale_lmbda = np.sqrt(lmbda_m2 - lmbda_mu**2)*0.7
    scale_c = np.sqrt(c_m2 - c_mu**2)*0.7
    scale_phi = np.sqrt(phi_m2 - phi_mu**2)*0.7
    scale_logsigmasq = np.sqrt(logsigmasq_m2 - logsigmasq_mu**2)*0.7
    
    return [scale_alpha, scale_lmbda, scale_c, scale_phi, scale_logsigmasq]

def push(theta_chain, theta, n) :
    alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain = theta_chain[:]
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    
    alpha_chain[n] = alpha
    lmbda_chain[n] = lmbda
    c_chain[n] = c
    phi_chain[n] = phi
    logsigmasq_chain[n] = logsigmasq
    
    return [alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain]

def log_prior(theta) :
    return 0 

def pMCMC(x_0, Y, theta_0, n_particles, n_mcmc, scale, power=1, adapt=True, start_adapt=0.2) :
    alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain, lls, theta_mu, theta_m2 = initialise(theta_0, n_mcmc)
    theta_chain = [alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain]
    
    theta_current = [alpha_chain[0], lmbda_chain[0], c_chain[0], phi_chain[0], logsigmasq_chain[0]]    
    lls[0] = bootstrap_PF(x_0, n_particles, theta_current, Y)
    accepted = 0
    last_jump = 0
    
    for n in trange(n_mcmc) :
        theta_proposed = propose(theta_current, scale)
        ll_proposed = bootstrap_PF(x_0, n_particles, theta_proposed, Y)
        log_prior_current, log_prior_proposed = log_prior(theta_current), log_prior(theta_proposed) 
        log_accept_prob = power*(ll_proposed-lls[n]) + (log_prior_proposed-log_prior_current)
        
        if np.log(npr.rand()) < log_accept_prob :
            lls[n+1] = ll_proposed
            theta_current = np.copy(theta_proposed)
            accepted += 1
            last_jump = n
        else :
            lls[n+1] = lls[n]
            if n - last_jump > 50 :
                lls[n+1] = bootstrap_PF(x_0, n_particles, theta_current, Y)
                
        if adapt :
            theta_mu, theta_m2 = update_moments(theta_mu, theta_m2, theta_current, n+1)
            if n >= int(n_mcmc*start_adapt) : 
                scale = adapt_scale(scale, theta_mu, theta_m2)
        
        theta_chain = push(theta_chain, theta_current, n+1)

    print(100*accepted/n_mcmc, "% acceptance rate")
    return theta_chain, scale

