# Calling libraries:
from __future__ import division
import numpy as np, math, numpy.random as npr, multiprocessing as mp, scipy, gc
from time import time
from scipy.stats import *
from tqdm import trange

gc.enable()

############################################ SIMULATE DATA ############################################

def simulate_data(theta, T) :
    npr.seed()
    scipy.random.seed()
    phi, sigma2, tau2 = theta[:]
    X, Y = np.zeros(T), np.zeros(T)
    X[0] = npr.randn()
    for t in np.arange(1,T) :
        X[t] = norm.rvs(loc=phi*X[t-1], scale=np.sqrt(sigma2))
    for t in range(T) :
        Y[t] = norm.rvs(loc=X[t], scale=np.sqrt(tau2)*np.exp(X[t]/2))
    return X, Y

####################################### PARTICLE FILTERING STUFF #######################################

def propagate(particles, theta) :
    phi, sigma2 = theta[0:2]
    return norm.rvs(loc=phi*particles, scale=np.sqrt(sigma2))

def potential(particles, y, theta) :
    phi, sigma2, tau2 = theta[:]
    return norm.pdf(x=y, loc=particles, scale=np.sqrt(tau2)*np.exp(particles/2))

def resample(particles, weights) :
    npr.seed()
    scipy.random.seed()
    n_particles = len(weights)
    particles = particles[npr.choice(a=n_particles, size=n_particles, p=weights/np.sum(weights))]
    weights = np.ones(n_particles)/n_particles
    return particles, weights

def bootstrap_PF_track(n_particles, theta, Y) :
    npr.seed()
    scipy.random.seed()
    T = len(Y)
    particles = np.zeros((T,n_particles))
    particles[0] = npr.randn(n_particles)
    weights = potential(particles[0], Y[0], theta)
    logNC = np.log(np.sum(weights))
    particles[0], weights = resample(particles[0], weights)
    
    for t in np.arange(1,T) :
        particles[t] = propagate(particles[t-1], theta)
        weights *= potential(particles[t], Y[t], theta)
        logNC += np.log(np.sum(weights))
        particles[t], weights = resample(particles[t], weights)
                
    return logNC, particles

def bootstrap_PF_mult(n_particles, theta, Y, m) :
    global f
    def f(n_particles) :
        return bootstrap_PF_track(n_particles, theta, Y)[0]
    pool = mp.Pool(min(m,10))
    result = pool.map(f, [n_particles for n_particles in [n_particles]*m])
    pool.close()
    gc.collect()
    return np.asarray(result)

def trans(theta) :
    return np.asarray([theta[0], np.exp(theta[1]), np.exp(theta[2])])


################################################ PRIOR ################################################

def log_prior(theta) :
    return -10**20*(np.abs(theta[0])>=1) + norm.logpdf(x=theta[1]) + norm.logpdf(x=theta[2])


############################################ PARTICLE MCMC ############################################

def pMCMC(Y, theta_0, n_particles, n_mcmc, scale, power=1, adapt=True, start_adapt=0.2) :
    
    theta_chain = np.zeros((n_mcmc+1,len(theta_0)))
    theta_chain[0] = np.asarray([theta_0[0], np.log(theta_0[1]), np.log(theta_0[2])])
    theta_mu = theta_chain[0]
    theta_m2 = theta_chain[0]**2
    lls = np.zeros((n_mcmc+1,power)) 
    lls[0] = bootstrap_PF_mult(n_particles, trans(theta_chain[0]), Y, power)
    scales = np.ones((n_mcmc+1,len(theta_0)))
    scales[:] = scale
    accepted = 0
    last_jump = 0

    for n in trange(n_mcmc) :
        theta_proposed = theta_chain[n] + scales[n]*npr.randn(3) 
        if np.abs(theta_proposed[0]) >= 1 :
            lls[n+1], theta_chain[n+1] = lls[n], theta_chain[n]
        else :
            ll_proposed = bootstrap_PF_mult(n_particles, trans(theta_proposed), Y, power)
            log_prior_current = log_prior(theta_chain[n])
            log_prior_proposed = log_prior(theta_proposed) 
            log_accept_prob = (np.sum(ll_proposed) - np.sum(lls[n])) \
                              + (log_prior_proposed - log_prior_current) \
                              + np.sum(theta_proposed[1:3] - theta_chain[n,1:3])
                              #+ np.log(np.prod(theta_proposed/theta_chain[n]))

            if np.log(npr.rand()) < log_accept_prob :
                lls[n+1], theta_chain[n+1] = ll_proposed, theta_proposed
                accepted += 1
                latest_jump = n
            else :
                lls[n+1], theta_chain[n+1] = lls[n], theta_chain[n]
                if n - last_jump > 50 :
                    lls[n+1] = bootstrap_PF_mult(n_particles, trans(theta_chain[n+1]), Y, power)

        theta_mu = ((n+1)*theta_mu + theta_chain[n+1])/(n+2)
        theta_m2 = ((n+1)*theta_m2 + theta_chain[n+1]**2)/(n+2)
        if adapt :
            if n >= int(n_mcmc*start_adapt) : 
                scales[n+1] = np.sqrt((theta_m2 - theta_mu**2))

    print(100*accepted/n_mcmc, "% acceptance rate")
    return theta_chain, scales


########################################## CODE FOR SANITY CHECK ##########################################

def log_prior_phi(phi) :
    return -10**20*(np.abs(phi)>=1)

def log_prior_log_sigma2(log_sigma2) :
    return (-1/2)*log_sigma2**2/10**2

def log_prior_log_tau2(log_tau2) :
    return (-1/2)*log_tau2**2/10**2

def pMCMC_phi(Y, theta_0, n_particles, n_mcmc, scale, power=1, adapt=True, start_adapt=0.2) :
    
    sigma2, tau2 = theta_0[1:3]
    phi_chain = np.zeros(n_mcmc+1)
    phi_chain[0] = theta_0[0]
    phi_mu, phi_m2  = phi_chain[0], phi_chain[0]**2
    lls = np.zeros((n_mcmc+1,power)) 
    lls[0] = bootstrap_PF_mult(n_particles, np.asarray([phi_chain[0], sigma2, tau2]), Y, power)
    scales = scale*np.ones(n_mcmc+1)
    accepted, last_jump = 0, 0

    for n in trange(n_mcmc) :
        phi_proposed = phi_chain[n] + scales[n]*npr.randn() 
        if np.abs(phi_proposed) >= 1 :
            lls[n+1], phi_chain[n+1] = lls[n], phi_chain[n]
        else :
            ll_proposed = bootstrap_PF_mult(n_particles, np.asarray([phi_proposed, sigma2, tau2]), Y, power)
            log_prior_current = log_prior_phi(phi_chain[n])
            log_prior_proposed = log_prior_phi(phi_proposed) 
            log_accept_prob = (np.sum(ll_proposed) - np.sum(lls[n])) \
                              + (log_prior_proposed - log_prior_current) \

            if np.log(npr.rand()) < log_accept_prob :
                lls[n+1], phi_chain[n+1] = ll_proposed, phi_proposed
                accepted += 1
                latest_jump = n
            else :
                lls[n+1], phi_chain[n+1] = lls[n], phi_chain[n]
                if n - last_jump > 50 :
                    lls[n+1] = bootstrap_PF_mult(n_particles, np.asarray([phi_chain[n+1], sigma2, tau2]), Y, power)

        phi_mu = ((n+1)*phi_mu + phi_chain[n+1])/(n+2)
        phi_m2 = ((n+1)*phi_m2 + phi_chain[n+1]**2)/(n+2)
        if adapt :
            if n >= int(n_mcmc*start_adapt) : 
                scales[n+1] = 3*np.sqrt((phi_m2 - phi_mu**2))

    print(100*accepted/n_mcmc, "% acceptance rate")
    return phi_chain, scales

def pMCMC_sigma2(Y, theta_0, n_particles, n_mcmc, scale, power=1, adapt=True, start_adapt=0.2) :
    
    phi, sigma2, tau2 = theta_0[:]
    log_sigma2_chain = np.zeros(n_mcmc+1)
    log_sigma2_chain[0] = np.log(sigma2)
    log_sigma2_mu, log_sigma2_m2 = log_sigma2_chain[0], log_sigma2_chain[0]**2
    lls = np.zeros((n_mcmc+1,power)) 
    lls[0] = bootstrap_PF_mult(n_particles, np.asarray([phi, np.exp(log_sigma2_chain[0]), tau2]), Y, power)
    scales = scale*np.ones(n_mcmc+1)
    accepted, last_jump = 0, 0

    for n in trange(n_mcmc) :
        log_sigma2_proposed = log_sigma2_chain[n] + scales[n]*npr.randn() 
        ll_proposed = bootstrap_PF_mult(n_particles, np.asarray([phi, np.exp(log_sigma2_proposed), tau2]), Y, power)
        log_prior_current = log_prior_log_sigma2(log_sigma2_chain[n])
        log_prior_proposed = log_prior_log_sigma2(log_sigma2_proposed) 
        log_accept_prob = (np.sum(ll_proposed) - np.sum(lls[n])) \
                          + (log_prior_proposed - log_prior_current) \
                          + log_sigma2_proposed - log_sigma2_chain[n]

        if np.log(npr.rand()) < log_accept_prob :
            lls[n+1], log_sigma2_chain[n+1] = ll_proposed, log_sigma2_proposed
            accepted += 1
            latest_jump = n
        else :
            lls[n+1], log_sigma2_chain[n+1] = lls[n], log_sigma2_chain[n]
            if n - last_jump > 50 :
                lls[n+1] = bootstrap_PF_mult(n_particles, np.asarray([phi, np.exp(log_sigma2_chain[n+1]), tau2]), Y, power)

        log_sigma2_mu = ((n+1)*log_sigma2_mu + log_sigma2_chain[n+1])/(n+2)
        log_sigma2_m2 = ((n+1)*log_sigma2_m2 + log_sigma2_chain[n+1]**2)/(n+2)
        if adapt :
            if n >= int(n_mcmc*start_adapt) : 
                scales[n+1] = 3*np.sqrt((log_sigma2_m2 - log_sigma2_mu**2))

    print(100*accepted/n_mcmc, "% acceptance rate")
    return log_sigma2_chain, scales

def pMCMC_phi_sigma2(Y, theta_0, n_particles, n_mcmc, scale, power=1, adapt=True, start_adapt=0.2) :
    
    phi, sigma2, tau2 = theta_0[:]
    phi_chain, log_sigma2_chain = np.zeros(n_mcmc+1), np.zeros(n_mcmc+1)
    phi_chain[0], log_sigma2_chain[0] = phi, np.log(sigma2)
    phi_mu, phi_m2  = phi_chain[0], phi_chain[0]**2
    log_sigma2_mu, log_sigma2_m2 = log_sigma2_chain[0], log_sigma2_chain[0]**2
    lls = np.zeros((n_mcmc+1,power)) 
    lls[0] = bootstrap_PF_mult(n_particles, np.asarray([phi_chain[0], np.exp(log_sigma2_chain[0]), tau2]), Y, power)
    scales = scale*np.ones(n_mcmc+1)
    accepted, last_jump = 0, 0

    for n in trange(n_mcmc) :
        phi_proposed = phi_chain[n] + scales[n,0]*npr.randn()
        log_sigma2_proposed = log_sigma2_chain[n] + scales[n,1]*npr.randn()
        if np.abs(phi_proposed) >= 1 :
            lls[n+1], phi_chain[n+1], log_sigma2_chain[n+1] = lls[n], phi_chain[n], log_sigma2_chain[n]
            
        ll_proposed = bootstrap_PF_mult(n_particles, np.asarray([phi_proposed, np.exp(log_sigma2_proposed), tau2]), Y, power)
        log_prior_current = log_prior_phi(phi_chain[n]) + log_prior_log_sigma2(log_sigma2_chain[n])
        log_prior_proposed = log_prior_phi(phi_proposed) + log_prior_log_sigma2(log_sigma2_proposed)
        log_accept_prob = (np.sum(ll_proposed) - np.sum(lls[n])) \
                          + (log_prior_proposed - log_prior_current) \
                          + log_sigma2_proposed - log_sigma2_chain[n]

        if np.log(npr.rand()) < log_accept_prob :
            lls[n+1], phi_chain[n+1], log_sigma2_chain[n+1] = ll_proposed, phi_proposed, log_sigma2_proposed
            accepted += 1
            latest_jump = n
        else :
            lls[n+1], phi_chain[n+1], log_sigma2_chain[n+1] = lls[n], phi_chain[n], log_sigma2_chain[n]
            if n - last_jump > 50 :
                lls[n+1] = bootstrap_PF_mult(n_particles, np.asarray([phi_chain[n+1], np.exp(log_sigma2_chain[n+1]), tau2]), Y, power)

        phi_mu = ((n+1)*phi_mu + phi_chain[n+1])/(n+2)
        phi_m2 = ((n+1)*phi_m2 + phi_chain[n+1]**2)/(n+2)
        log_sigma2_mu = ((n+1)*log_sigma2_mu + log_sigma2_chain[n+1])/(n+2)
        log_sigma2_m2 = ((n+1)*log_sigma2_m2 + log_sigma2_chain[n+1]**2)/(n+2)
        if adapt :
            if n >= int(n_mcmc*start_adapt) : 
                scales[n+1,0] = 3*np.sqrt((phi_m2 - phi_mu**2))
                scales[n+1,1] = 3*np.sqrt((log_sigma2_m2 - log_sigma2_mu**2))

    print(100*accepted/n_mcmc, "% acceptance rate")
    return phi_chain, log_sigma2_chain, scales

def pMCMC_tau2(Y, theta_0, n_particles, n_mcmc, scale, power=1, adapt=True, start_adapt=0.2) :
    
    phi, sigma2, tau2 = theta_0[:]
    log_tau2_chain = np.zeros(n_mcmc+1)
    log_tau2_chain[0] = np.log(sigma2)
    log_tau2_mu, log_tau2_m2 = log_tau2_chain[0], log_tau2_chain[0]**2
    lls = np.zeros((n_mcmc+1,power)) 
    lls[0] = bootstrap_PF_mult(n_particles, np.asarray([phi, sigma2, np.exp(log_tau2_chain[0])]), Y, power)
    scales = scale*np.ones(n_mcmc+1)
    accepted, last_jump = 0, 0

    for n in trange(n_mcmc) :
        log_tau2_proposed = log_tau2_chain[n] + scales[n]*npr.randn() 
        ll_proposed = bootstrap_PF_mult(n_particles, np.asarray([phi, sigma2, np.exp(log_tau2_proposed)]), Y, power)
        log_prior_current = log_prior_log_tau2(log_tau2_chain[n])
        log_prior_proposed = log_prior_log_tau2(log_tau2_proposed) 
        log_accept_prob = (np.sum(ll_proposed) - np.sum(lls[n])) \
                          + (log_prior_proposed - log_prior_current) \
                          + log_tau2_proposed - log_tau2_chain[n]

        if np.log(npr.rand()) < log_accept_prob :
            lls[n+1], log_tau2_chain[n+1] = ll_proposed, log_tau2_proposed
            accepted += 1
            latest_jump = n
        else :
            lls[n+1], log_tau2_chain[n+1] = lls[n], log_tau2_chain[n]
            if n - last_jump > 50 :
                lls[n+1] = bootstrap_PF_mult(n_particles, np.asarray([phi, sigma2, np.exp(log_tau2_chain[n+1])]), Y, power)

        log_tau2_mu = ((n+1)*log_tau2_mu + log_tau2_chain[n+1])/(n+2)
        log_tau2_m2 = ((n+1)*log_tau2_m2 + log_tau2_chain[n+1]**2)/(n+2)
        if adapt :
            if n >= int(n_mcmc*start_adapt) : 
                scales[n+1] = 3*np.sqrt((log_tau2_m2 - log_tau2_mu**2))

    print(100*accepted/n_mcmc, "% acceptance rate")
    return log_tau2_chain, scales

