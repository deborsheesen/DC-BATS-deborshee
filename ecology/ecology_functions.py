from __future__ import division
import numpy as np, time, matplotlib.pyplot as plt, math, pandas, numpy.random as npr, multiprocessing as mp, copy, gc
from time import time
from scipy.stats import *
from tqdm import trange
from numpy.matlib import repmat
import scipy
gc.enable()

def propagate(particles, theta) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    sigma = np.exp(logsigmasq/2)
    return c + phi*particles + sigma*npr.randn(*np.shape(particles)) 

def simulate_data(x_0, T, J, theta) :   # I = no. of locations, J = no. of species, K = no. of latent factors
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    I, K = np.shape(x_0)
    Y = np.zeros((T,I,J)).astype(int)
    X = np.zeros((T+1,I,K))
    X[0] = x_0

    for t in range(T) :
        X[t+1] = propagate(X[t], theta)
        for j in range(J) :
            reg = alpha[j] + lmbda[j,:].dot(X[t+1].transpose())
            probs = 1/(1+np.exp(-reg))
            Y[t,:,j] = npr.binomial(n=1, p=probs).transpose()
    
    return Y, X

#####################################################################################################################
#################################------    BOOTSTRAP PARTICLE FILTER   ------########################################
#####################################################################################################################

# Implementing ALgorithm 1 from https://arxiv.org/pdf/1901.10568.pdf for ecology model

def adaptive_resample(weights, particles) :
    weights /= np.sum(weights)
    ESS = 1/np.sum(weights**2)
    n_particles = len(weights)
    idx_resampled = np.arange(n_particles)
    if ESS < n_particles/2 :
        particles = particles[:,:,npr.choice(a=n_particles,size=n_particles,p=weights)]
        weights = np.ones(n_particles)/n_particles
    return weights, particles 

def potential(y, particles, theta) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    I, K, n_particles = np.shape(particles)
    J = np.shape(y)[-1]
    #reg = np.zeros((J,n_particles,I))
    #for j in range(J) :
        #reg[j] = alpha[j] + lmbda[j].dot(particles.transpose())
    reg = np.swapaxes(np.reshape(alpha, [J,1,1]) + np.sum(np.reshape(particles, [1,I,K,n_particles])*np.reshape(lmbda,[J,1,K,1]),2),1,2)
    prob = (1/(1+np.exp(-reg))).transpose()
    return np.prod(np.swapaxes(prob,1,2)**np.reshape(y, (I,J,1))*(1-np.swapaxes(prob,1,2))**(1-np.reshape(y, (I,J,1))), (0,1))

# --------------------------------------------- Track particles ------------------------------------------------- #

def bootstrap_PF_track(Y, x_0, n_particles, theta) :
    
    np.random.seed()
    scipy.random.seed()
    
    T, I, J = np.shape(Y)
    K = np.shape(x_0)[-1]
    particles = np.zeros((T+1,*np.shape(x_0),n_particles))   # I = number of locations, K = dimension at each location
    for n in range(n_particles) :
        particles[0,:,:,n] = x_0
    weights = np.ones(n_particles)/n_particles 
    logNC = 0
    
    for t in range(T) :
        particles[t+1] = propagate(particles[t], theta)
        incremental_weights = potential(Y[t], particles[t+1], theta)
        weights = weights*incremental_weights
        logNC += np.log(np.sum(weights))
        resampled_idx = npr.choice(a=n_particles, size=n_particles, p=weights/np.sum(weights))
        part = particles[t+1]
        particles[t+1] = part[:,:,resampled_idx]
        weights = np.ones(n_particles)/n_particles
    return logNC, particles, weights

# ----------------------------- Functions to calculate gradient using Fisher's identity ----------------------------- #

# these have to return vectors
def alpha_grad(y, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    I, K, n_particles = np.shape(propagated_particles)
    J = len(alpha)
    reg = np.reshape(alpha,[J,1,1]) + np.sum(np.reshape(lmbda,[J,1,K,1])*np.reshape(propagated_particles,[1,I,K,n_particles]),2)
    prob = 1/(1+np.exp(-reg))
    grad = np.transpose(np.sum(prob - np.reshape(y.transpose(), [J,I,1]),1))
    return grad

def lmbda_grad(y, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    I, K, n_particles = np.shape(propagated_particles)
    J = len(alpha)
    reg = np.reshape(alpha,[J,1,1,1]) + np.reshape(np.sum(np.reshape(lmbda,[J,1,K,1])*np.reshape(propagated_particles,[1,I,K,n_particles]),2), [J,I,1,n_particles])
    prob = 1/(1+np.exp(-reg))
    grad = np.sum((prob-np.reshape(y.transpose(),[J,I,1,1]))*np.reshape(propagated_particles, [1,I,K,n_particles]),1)
    return np.swapaxes(grad.transpose(),1,2)

def c_grad(y, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    sigmasq = np.exp(logsigmasq)
    I, K, n_particles = np.shape(particles)
    return -1/(sigmasq)*(c*(I*K)**2 + np.sum(phi*particles-propagated_particles, (0,1)))

def phi_grad(y, theta, propagated_particles, particles) : 
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    sigmasq = np.exp(logsigmasq)
    return -1/(sigmasq)*(phi*np.sum(particles**2 - c*particles + particles*propagated_particles, (0,1)))

def logsigmasq_grad(y, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    sigmasq = np.exp(logsigmasq)
    return 1/(2*sigmasq)*np.sum((propagated_particles-(c+phi*particles))**2, (0,1))

def update_gradient(grad, y, propagated_particles, resampled_particles, resampled_idx, theta) :
    grad_alpha, grad_lmbda, grad_c, grad_phi, grad_logsigmasq = grad[:]
    grad_alpha[:] = grad_alpha[resampled_idx] + alpha_grad(y, theta, propagated_particles, resampled_particles)
    grad_lmbda[:] = grad_lmbda[resampled_idx] + lmbda_grad(y, theta, propagated_particles, resampled_particles)
    grad_c[:] = grad_c[resampled_idx] + c_grad(y, theta, propagated_particles, resampled_particles)
    grad_phi[:] = grad_phi[resampled_idx] + phi_grad(y, theta, propagated_particles, resampled_particles)
    grad_logsigmasq[:] = grad_logsigmasq[resampled_idx] + logsigmasq_grad(y, theta, propagated_particles, resampled_particles)
    return [grad_alpha, grad_lmbda, grad_c, grad_phi, grad_logsigmasq]

def weigh_grad(grad, weights) :
    n_particles = len(weights) 
    grad_est_alpha = np.sum(grad[0]*np.reshape(weights, [n_particles,1]), 0)/np.sum(weights)
    grad_est_lmbda = np.sum(grad[1]*np.reshape(weights, [n_particles,1,1]), 0)/np.sum(weights)
    grad_est_c = np.sum(grad[2]*weights)/np.sum(weights)
    grad_est_phi = np.sum(grad[3]*weights)/np.sum(weights)
    grad_est_logsigmasq = np.sum(grad[4]*weights)/np.sum(weights)
    return [grad_est_alpha, grad_est_lmbda, grad_est_c, grad_est_phi, grad_est_logsigmasq]

def bootstrap_PF_grad(Y, x_0, n_particles, theta, calc_grad=True) :
    
    np.random.seed()
    scipy.random.seed()
    
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    if calc_grad : 
        grad = [np.zeros((n_particles,*np.shape(alpha))), np.zeros((n_particles,*np.shape(lmbda))), \
                np.zeros(n_particles), np.zeros(n_particles), np.zeros(n_particles)]
    
    T, I, J = np.shape(Y)
    I, K = np.shape(x_0)
    particles, weights, logNC = np.zeros((I,K,n_particles)), np.ones(n_particles)/n_particles, 0.
    for n in range(n_particles) : particles[:,:,n] = x_0
    for t in range(T) :
        resampled_idx = npr.choice(a=n_particles, size=n_particles, p=weights/np.sum(weights))
        resampled_particles = particles[:,:,resampled_idx]
        propagated_particles = propagate(resampled_particles, theta)
        #incremental_weights = potential(Y[t], propagated_particles, theta)
        #weights = weights*incremental_weights
        weights = potential(Y[t], propagated_particles, theta)
        logNC += np.log(np.mean(weights))
        
        if calc_grad : 
            grad = update_gradient(grad, Y[t], propagated_particles, resampled_particles, resampled_idx, theta)
        particles = np.copy(propagated_particles)
    
    weights /= np.sum(weights) # normalise weights at the end
    
    if calc_grad : 
        return logNC, weigh_grad(grad, weights)
    else :
        return logNC, 0 


#####################################################################################################################
#########################################      BLOCK PARTICLE FILTER     ############################################
#####################################################################################################################

def alpha_grad_blockPF(y, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    n_particles, I, K = np.shape(propagated_particles)
    J = len(alpha)
    grad = np.zeros((n_particles,J,I))
    reg = np.reshape(alpha, [J,1,1]) + np.sum(np.reshape(lmbda.transpose(), [K,J,1,1])*np.reshape(propagated_particles.transpose(), [K,1,I,n_particles]),0)
    prob = np.swapaxes((np.exp(reg)/(1+np.exp(reg))).transpose(),1,2)
    grad = prob - np.reshape(y.transpose(), [1,J,I])
    return grad

def lmbda_grad_blockPF(y, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    n_particles, I, K = np.shape(propagated_particles)
    J = len(alpha)
    reg = np.reshape(alpha, [J,1,1]) + np.sum(np.reshape(lmbda.transpose(), [K,J,1,1])\
                                          *np.reshape(propagated_particles.transpose(), [K,1,I,n_particles]),0)
    prob = 1/(1+np.exp(-reg))
    grad = np.reshape(prob.transpose(), [n_particles,I,1,J])*np.reshape(propagated_particles, [n_particles,I,K,1]) \
            - np.reshape(y, [1,I,1,J])*np.reshape(propagated_particles, [n_particles,I,K,1]) 
    return np.swapaxes(grad, 1,3)

def c_grad_blockPF(y, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    sigmasq = np.exp(logsigmasq)
    n_particles, I, K = np.shape(particles)
    return -1/(sigmasq)*(c*(I*K)**2 + np.sum(phi*particles-propagated_particles,-1))

def phi_grad_blockPF(y, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    sigmasq = np.exp(logsigmasq)
    return -1/(sigmasq)*(phi*np.sum(particles**2 - c*particles + particles*propagated_particles, -1))

def logsigmasq_grad_blockPF(y, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    sigmasq = np.exp(logsigmasq)
    return 1/(2*sigmasq)*np.sum((propagated_particles-(c+phi*particles))**2, -1)

def update_gradient_blockPF_old(grad, y, propagated_particles, particles, resampled_idx, theta) :
    grad_alpha, grad_lmbda, grad_c, grad_phi, grad_logsigmasq = grad[:]
    I = np.shape(grad_alpha)[-1]
    
    for i in range(I) : # this loop is slow..
        grad_alpha[:,:,i] = grad_alpha[resampled_idx[:,i],:,i] \
                            + alpha_grad_blockPF(y, theta, propagated_particles, particles[resampled_idx[:,i]])[:,:,i]
        grad_lmbda[:,:,:,i] = grad_lmbda[resampled_idx[:,i],:,:,i] \
                            + lmbda_grad_blockPF(y, theta, propagated_particles, particles[resampled_idx[:,i]])[:,:,:,i]
        grad_c[:,i] = grad_c[resampled_idx[:,i],i] \
                      + c_grad_blockPF(y, theta, propagated_particles, particles[resampled_idx[:,i]])[:,i]
        grad_phi[:,i] = grad_phi[resampled_idx[:,i],i] \
                        + phi_grad_blockPF(y, theta, propagated_particles, particles[resampled_idx[:,i]])[:,i]
        grad_logsigmasq[:,i] = grad_logsigmasq[resampled_idx[:,i],i] \
                               + logsigmasq_grad_blockPF(y, theta, propagated_particles, particles[resampled_idx[:,i]])[:,i]
    
    return [grad_alpha, grad_lmbda, grad_c, grad_phi, grad_logsigmasq]


def update_gradient_blockPF(grad, y, propagated_particles, resampled_particles, resampled_idx, theta) :
    grad_alpha, grad_lmbda, grad_c, grad_phi, grad_logsigmasq = grad[:]
    n_particles, I, K = np.shape(propagated_particles)
    #resampled_particles = np.zeros((n_particles,I,K))
    #for i in range(I) :
    #    resampled_particles[:,i] = particles[resampled_idx[:,i],i]
    alpha_grad = alpha_grad_blockPF(y, theta, propagated_particles, resampled_particles)
    lmbda_grad = lmbda_grad_blockPF(y, theta, propagated_particles, resampled_particles)
    c_grad = c_grad_blockPF(y, theta, propagated_particles, resampled_particles)
    phi_grad = phi_grad_blockPF(y, theta, propagated_particles, resampled_particles)
    logsigmasq_grad = logsigmasq_grad_blockPF(y, theta, propagated_particles, resampled_particles)
    
    for i in range(I) : # this loop is slow..
        grad_alpha[:,:,i] = grad_alpha[resampled_idx[:,i],:,i] + alpha_grad[:,:,i]
        grad_lmbda[:,:,:,i] = grad_lmbda[resampled_idx[:,i],:,:,i] + lmbda_grad[:,:,:,i]
        grad_c[:,i] = grad_c[resampled_idx[:,i],i] + c_grad[:,i]
        grad_phi[:,i] = grad_phi[resampled_idx[:,i],i] + phi_grad[:,i]
        grad_logsigmasq[:,i] = grad_logsigmasq[resampled_idx[:,i],i] + logsigmasq_grad[:,i]
    
    return [grad_alpha, grad_lmbda, grad_c, grad_phi, grad_logsigmasq]

def local_potential(y, particles, theta) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    n_particles, I, K = np.shape(particles)
    J = np.shape(y)[1]
    #reg = np.zeros((J,I,n_particles))
    #for j in range(J) :
    #    reg[j] = alpha[j] + np.sum(np.reshape(lmbda[j],(K,1,1))*particles.transpose(), 0)
    reg = np.reshape(alpha, [J,1,1]) + np.swapaxes(np.sum(np.reshape(lmbda.transpose(), [K,1,J,1]) * \
                     np.reshape(particles.transpose(), [K,I,1,n_particles]),0), 0,1)
    prob = 1/(1+np.exp(-reg))
    yy = np.reshape(y.transpose(), (J,I,1))
    return np.prod((prob**yy)*(1-prob)**(1-yy),0).transpose()

# this stores particle paths and computes gradient
def block_PF_track(Y, x_0, n_particles, theta, calc_grad=True) : # I = number of locations, K = dimension at each location
    
    np.random.seed()
    scipy.random.seed()
    
    T, I, J = np.shape(Y)
    K = np.shape(x_0)[-1]
    particles = np.zeros((T+1,n_particles,I,K))  
    resampled_particles = np.zeros((n_particles,I,K))  
    particles[0] = x_0
    local_weights = np.ones((n_particles,I))/n_particles 
    resampled_idx = np.zeros((n_particles,I)).astype(int)
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    logNC = 0

    if calc_grad : 
        grad = [np.zeros((n_particles,*np.shape(alpha),I)), np.zeros((n_particles,*np.shape(lmbda),I)), \
                np.zeros((n_particles,I)), np.zeros((n_particles,I)), np.zeros((n_particles,I))]

    for t in range(T) :
        # resampled_idx = (weights.cumsum(0) > npr.rand(n_particles,weights.shape[1])[:,None]).argmax(1) #this is actually slower
        for i in range(I) :
            resampled_idx[:,i] = npr.choice(a=n_particles, size=n_particles, p=local_weights[:,i]/np.sum(local_weights[:,i]))
            particles[t,:,i] = particles[t,resampled_idx[:,i],i]
            resampled_particles[:,i] = particles[t,resampled_idx[:,i],i]
        particles[t+1] = propagate(resampled_particles, theta)  
        local_weights = local_potential(Y[t], particles[t+1], theta)
        logNC += np.log(np.mean(np.prod(local_weights,1)))
        
        if calc_grad : 
            grad = update_gradient_blockPF(grad, Y[t], particles[t+1], particles[t], resampled_idx, theta) 
        
    local_weights /= np.sum(local_weights,0) # normalise weights at the end 
    
    if calc_grad :
        grad_est_alpha = np.sum(np.swapaxes(grad[0],1,2)*np.reshape(local_weights, (*np.shape(local_weights),1)), (0,1))
        grad_est_lmbda = np.sum(np.swapaxes(np.swapaxes(grad[1],1,3), 2,3)
                                *np.reshape(local_weights, (*np.shape(local_weights),1,1)),(0,1))
        grad_est_c = np.sum(grad[2]*local_weights)
        grad_est_phi = np.sum(grad[3]*local_weights)
        grad_est_logsigmasq = np.sum(grad[4]*local_weights)
        grad = [grad_est_alpha, grad_est_lmbda, grad_est_c, grad_est_phi, grad_est_logsigmasq]        
        return logNC, grad, particles, local_weights
    else :
        return logNC, 0, particles, local_weights
    
# don't track particles
def block_PF(Y, x_0, n_particles, theta, calc_grad=True) : # I = number of locations, K = dimension at each location
    
    np.random.seed()
    scipy.random.seed()
    
    T, I, J = np.shape(Y)
    K = np.shape(x_0)[-1]
    particles = np.zeros((n_particles,I,K))  
    resampled_particles = np.zeros((n_particles,I,K))  
    particles[:] = x_0
    local_weights = np.ones((n_particles,I))/n_particles 
    resampled_idx = np.zeros((n_particles,I)).astype(int)
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    logNC = 0

    if calc_grad : 
        grad = [np.zeros((n_particles,*np.shape(alpha),I)), np.zeros((n_particles,*np.shape(lmbda),I)), \
                np.zeros((n_particles,I)), np.zeros((n_particles,I)), np.zeros((n_particles,I))]

    for t in range(T) :
        for i in range(I) :
            resampled_idx[:,i] = npr.choice(a=n_particles, size=n_particles, p=local_weights[:,i]/np.sum(local_weights[:,i]))
            resampled_particles[:,i] = particles[resampled_idx[:,i],i]
        propagated_particles = propagate(resampled_particles, theta)  
        local_weights = local_potential(Y[t], propagated_particles, theta)
        logNC += np.log(np.mean(np.prod(local_weights,1)))
        
        if calc_grad : 
            grad = update_gradient_blockPF(grad, Y[t], propagated_particles, resampled_particles, resampled_idx, theta)
        particles = np.copy(propagated_particles)
        
    local_weights /= np.sum(local_weights,0) # normalise weights at the end 
    
    if calc_grad :
        grad_est_alpha = np.sum(np.swapaxes(grad[0],1,2)*np.reshape(local_weights, (*np.shape(local_weights),1)), (0,1))
        grad_est_lmbda = np.sum(np.swapaxes(np.swapaxes(grad[1],1,3), 2,3)
                                *np.reshape(local_weights, (*np.shape(local_weights),1,1)),(0,1))
        grad_est_c = np.sum(grad[2]*local_weights)
        grad_est_phi = np.sum(grad[3]*local_weights)
        grad_est_logsigmasq = np.sum(grad[4]*local_weights)
        grad = [grad_est_alpha, grad_est_lmbda, grad_est_c, grad_est_phi, grad_est_logsigmasq]        
        return logNC, grad
    else :
        return logNC

#####################################################################################################################
###########################################      PSEUDO MARGINAL MCMC     ###########################################
#####################################################################################################################


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
    return -(1/2)*np.sum([np.linalg.norm(theta[i])**2 for i in range(5)])

#####################################  ADAPTIVE RANDOM WALK METROPOLIS-HASTINGS  #####################################

def initialise(theta_0, n_mcmc) :
    alpha, lmbda, c, phi, logsigmasq = theta_0[:]
    grad = [np.zeros(*np.shape(alpha)), np.zeros(np.shape(lmbda)), 0, 0, 0]
    
    alpha_chain = np.zeros((n_mcmc+1,*np.shape(alpha)))
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

def propose_RW(theta, scale, update) :
    np.random.seed()
    scipy.random.seed()
    
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    scale_alpha, scale_lmbda, scale_c, scale_phi, scale_logsigmasq = scale[:]
    
    alpha_prop = npr.multivariate_normal(alpha, scale_alpha)
    lmbda_prop = npr.multivariate_normal(lmbda.reshape(np.prod(np.shape(lmbda))), scale_lmbda).reshape(*np.shape(lmbda))
    c_prop = c + update[2]*scale_c*npr.randn()
    phi_prop = phi + update[3]*scale_phi*npr.randn()
    logsigmasq_prop = logsigmasq + update[4]*scale_logsigmasq*npr.randn()
    
    return [alpha_prop, lmbda_prop, c_prop, phi_prop, logsigmasq_prop]

def update_moments(theta_mu, theta_m2, theta_new, n) :
    theta_mu = [(n*theta_mu[i] + theta_new[i])/(n+1) for i in range(5)]
    theta_m2 = [(n*theta_m2[i] + theta_new[i]**2)/(n+1) for i in range(5)]
    return theta_mu, theta_m2

def adapt_scale(scale, theta_mu, theta_m2) :
    return [0.7*np.sqrt(theta_m2[i]-theta_mu[i]**2) for i in range(5)]
    
def pMCMC_RW(x_0, Y, theta_0, n_particles, n_mcmc, scale, update, power=1, adapt=True, start_adapt=0.2) :
    
    np.random.seed()
    scipy.random.seed()
    
    alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain, lls, theta_mu, theta_m2 = initialise(theta_0, n_mcmc)
    theta_chain = [alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain]
    
    theta_curr = [alpha_chain[0], lmbda_chain[0], c_chain[0], phi_chain[0], logsigmasq_chain[0]]    
    lls[0] = block_PF(Y, x_0, n_particles, theta_curr, calc_grad=False)
    accepted = 0
    last_jump = 0
    
    accept_probs = np.zeros(n_mcmc)
    
    for n in trange(n_mcmc) :
        theta_prop = propose_RW(theta_curr, scale, update)
        ll_prop = block_PF(Y, x_0, n_particles, theta_prop, calc_grad=False)
        log_prior_curr, log_prior_prop = log_prior(theta_curr), log_prior(theta_prop) 
        log_accept_prob = power*(ll_prop-lls[n]) + (log_prior_prop-log_prior_curr)
        accept_probs[n] = np.exp(log_accept_prob)
        
        if np.log(npr.rand()) < log_accept_prob :
            lls[n+1] = ll_prop
            theta_curr = np.copy(theta_prop)
            accepted += 1
            last_jump = n
        else :
            lls[n+1] = lls[n]
            if n - last_jump > 50 :
                lls[n+1] = block_PF(Y, x_0, n_particles, theta_curr, calc_grad=False)
        if adapt :
            theta_mu, theta_m2 = update_moments(theta_mu, theta_m2, theta_curr, n+1)
            if n >= int(n_mcmc*start_adapt) : 
                scale = adapt_scale(scale, theta_mu, theta_m2)
        
        theta_chain = push(theta_chain, theta_curr, n+1)

    print(100*accepted/n_mcmc, "% acceptance rate")
    return theta_chain, scale, accept_probs

############################################# MULTIPLE #############################################

def block_PF_mult(Y, x_0, n_particles, theta, m) :
    global f
    def f(n_particles) :
        return block_PF(Y, x_0, n_particles, theta, calc_grad=False)
    pool = mp.Pool(min(m,10))
    result = pool.map(f, [n_particles for n_particles in [n_particles]*m])
    pool.close()
    gc.collect()
    return np.asarray(result)

def pMCMC_RW_mult(x_0, Y, theta_0, n_particles, n_mcmc, scale, update, power=1, adapt=True, start_adapt=0.2) :
    
    np.random.seed()
    scipy.random.seed()
    
    alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain, lls, theta_mu, theta_m2 = initialise(theta_0, n_mcmc)
    theta_chain = [alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain]
    lls = np.zeros((n_mcmc+1,power))
    
    theta_curr = [alpha_chain[0], lmbda_chain[0], c_chain[0], phi_chain[0], logsigmasq_chain[0]]   
    lls[0] = block_PF_mult(Y, x_0, n_particles, theta_curr, power)
    accepted = 0
    last_jump = 0
    
    accept_probs = np.zeros(n_mcmc)
    
    for n in trange(n_mcmc) :
        theta_prop = propose_RW(theta_curr, scale, update)
        ll_prop = block_PF_mult(Y, x_0, n_particles, theta_prop, power)
        log_prior_curr, log_prior_prop = log_prior(theta_curr), log_prior(theta_prop) 
        log_accept_prob = np.sum(ll_prop-lls[n]) + (log_prior_prop-log_prior_curr)
        accept_probs[n] = np.exp(log_accept_prob)
        
        if np.log(npr.rand()) < log_accept_prob :
            lls[n+1] = ll_prop
            theta_curr = np.copy(theta_prop)
            accepted += 1
            last_jump = n
        else :
            lls[n+1] = lls[n]
            if n - last_jump > 50 :
                lls[n+1] = block_PF_mult(Y, x_0, n_particles, theta_curr, power)
        if adapt :
            theta_mu, theta_m2 = update_moments(theta_mu, theta_m2, theta_curr, n+1)
            if n >= int(n_mcmc*start_adapt) : 
                scale = adapt_scale(scale, theta_mu, theta_m2)
        
        theta_chain = push(theta_chain, theta_curr, n+1)

    print(100*accepted/n_mcmc, "% acceptance rate")
    return theta_chain, scale, accept_probs

#####################################################################################################################
############################################### PSEUDO MARGINAL MALA ################################################
#####################################################################################################################

############################################## Without using autograd ###############################################

def propose_MALA(theta, grad, tau, update=None) :
    np.random.seed()
    scipy.random.seed()
    
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    grad_alpha, grad_lmbda, grad_c, grad_phi, grad_logsigmasq = grad[:]
    if update == None :
        update = np.ones(5)
    
    alpha_prop = alpha + update[0]*(tau[0]*grad_alpha + np.sqrt(2*tau[0])*npr.randn(*np.shape(alpha)))
    lmbda_prop = lmbda + update[1]*(tau[1]*grad_lmbda + np.sqrt(2*tau[1])*npr.randn(*np.shape(lmbda)))
    c_prop = c + update[2]*(tau[2]*grad_c + np.sqrt(2*tau[2])*npr.randn())
    phi_prop = phi + update[3]*(tau[3]*grad_phi + np.sqrt(2*tau[3])*npr.randn())
    logsigmasq_prop = logsigmasq + update[4]*(tau[4]*grad_logsigmasq + np.sqrt(2*tau[4])*npr.randn())
    
    return [alpha_prop, lmbda_prop, c_prop, phi_prop, logsigmasq_prop]

def transition_prob_MALA(theta_curr, theta_prop, ll_curr, ll_prop, grad_curr, grad_prop, tau, power, update) :
    log_prior_curr, log_prior_prop = log_prior(theta_curr), log_prior(theta_prop) 
    a = power*(ll_prop-ll_curr) + (log_prior_prop-log_prior_curr)
    b1 = -np.sum([np.linalg.norm(theta_curr[i] - theta_prop[i] - update[i]*tau[i]*grad_curr[i])**2/(4*tau[i]) for i in range(5)])
    b2 = -np.sum([np.linalg.norm(theta_prop[i] - theta_curr[i] - update[i]*tau[i]*grad_prop[i])**2/(4*tau[i]) for i in range(5)])
    return a + b1 - b2

def pMCMC_MALA(x_0, Y, theta_0, n_particles, n_mcmc, tau, update=None, power=1) :
    
    np.random.seed()
    scipy.random.seed()
    
    alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain, lls, theta_mu, theta_m2 = initialise(theta_0, n_mcmc)
    theta_chain = [alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain]
    
    theta_curr = [alpha_chain[0], lmbda_chain[0], c_chain[0], phi_chain[0], logsigmasq_chain[0]]    
    ll_curr, grad_curr = bootstrap_PF_grad(Y, x_0, n_particles, theta_curr, Y)
    accepted = 0
    last_jump = 0
    
    accept_probs = np.zeros(n_mcmc)
    
    for n in trange(n_mcmc) :
        theta_prop = propose_MALA(theta_curr, grad_curr, tau, update)
        ll_prop, grad_prop = bootstrap_PF_grad(Y, x_0, n_particles, theta_prop)
        
        log_accept_prob = transition_prob_MALA(theta_curr, theta_prop, ll_curr, ll_prop, grad_curr, grad_prop, tau, power) 
        accept_probs[n] = np.exp(log_accept_prob)
        
        if np.log(npr.rand()) < log_accept_prob :
            ll_curr = np.copy(ll_prop)
            theta_curr = np.copy(theta_prop)
            grad_curr = np.copy(grad_prop)
            accepted += 1
            last_jump = n
        else :
            if n - last_jump > 50 :
                ll_curr, grad_curr = bootstrap_PF_grad(Y, x_0, n_particles, theta_curr)
        
        theta_chain = push(theta_chain, theta_curr, n+1)

    print(100*accepted/n_mcmc, "% acceptance rate")
    return theta_chain, accept_probs

def pMCMC_MALA_blockPF(Y, x_0, n_particles, theta_0, n_mcmc, tau, update=None, power=1) :
    
    np.random.seed()
    scipy.random.seed()
    
    alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain, lls, theta_mu, theta_m2 = initialise(theta_0, n_mcmc)
    theta_chain = [alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain]
    
    theta_curr = [alpha_chain[0], lmbda_chain[0], c_chain[0], phi_chain[0], logsigmasq_chain[0]]    
    ll_curr, grad_curr = block_PF(Y, x_0, n_particles, theta_curr, calc_grad=True)
    accepted = 0
    last_jump = 0
    
    accept_probs = np.zeros(n_mcmc)
    
    for n in trange(n_mcmc) :
        theta_prop = propose_MALA(theta_curr, grad_curr, tau, update)
        ll_prop, grad_prop = block_PF(Y, x_0, n_particles, theta_prop, calc_grad=True)
        log_accept_prob = transition_prob_MALA(theta_curr, theta_prop, ll_curr, ll_prop, grad_curr, grad_prop, tau, power, update) 
        accept_probs[n] = np.exp(log_accept_prob)
        
        if np.log(npr.rand()) < log_accept_prob :
            ll_curr = np.copy(ll_prop)
            theta_curr = np.copy(theta_prop)
            grad_curr = np.copy(grad_prop)
            accepted += 1
            last_jump = n
        else :
            if n - last_jump > 50 :
                ll_curr, grad_curr = block_PF(Y, x_0, n_particles, theta_curr, calc_grad=True)
        
        theta_chain = push(theta_chain, theta_curr, n+1)

    print(100*accepted/n_mcmc, "% acceptance rate")
    return theta_chain, accept_probs



def plot_theta_trajectory(theta_mcmc) :
    J, K = np.shape(theta_mcmc[1])[1], np.shape(theta_mcmc[1])[2]
    plt.rcParams['figure.figsize'] = (18, 2.5)
    titles = [r"$\alpha$", "$c$", r"$\phi$", r"$\log (\sigma^2)$"]
    for (i,j) in enumerate([0,2,3,4]) :
        plt.subplot(1,4,i+1)
        plt.plot(theta_mcmc[j])
        plt.title(titles[i], fontsize=15)
        plt.grid(True)
    plt.show()

    plt.rcParams['figure.figsize'] = (4*K, 1.2*J)
    for j in range(J) :
        for k in range(K) :
            idx = j*K + k + 1
            plt.subplot(J,K,idx)
            plt.plot(theta_mcmc[1][:,j,k])
            plt.grid(True)
            if j < (J-1) : plt.xticks(alpha=0)
    plt.subplots_adjust(hspace=0)
    #plt.suptitle(r"$\lambda$", fontsize=15)
    plt.show()

############################################ FOR SANITY CHECK  ############################################

def get_grads(Y, x_0, n_particles, theta, Tmax, Imax, Jmax, Kmax, rep=500) :
    Y = Y[:Tmax,:Imax,:Jmax]
    x_0 = x_0[:Imax,:Kmax]
    
    theta = [theta[0][:Jmax], theta[1][:Jmax,:Kmax], theta[2], theta[3], theta[4]]
    
    global f1, f2
    def f1(n_particles) :
        return block_PF(Y, x_0, n_particles, theta, calc_grad=True)
    def f2(n_particles) :
        return bootstrap_PF_grad(Y, x_0, n_particles, theta, calc_grad=True)

    pool = mp.Pool(10)
    results1 = pool.map(f1, [n_particles for n_particles in [n_particles]*rep])
    results2 = pool.map(f2, [n_particles for n_particles in [n_particles]*rep])
    pool.close()

    I, K = np.shape(x_0)
    J = np.shape(Y)[-1]
    logNC = np.zeros((rep,2))
    alpha_grad, lmbda_grad = np.zeros((rep,2,J)), np.zeros((rep,2,J,K))
    c_grad, phi_grad, logsigmasq_grad = np.zeros((rep,2)), np.zeros((rep,2)), np.zeros((rep,2))

    for r in range(rep) :
        alpha_grad[r,0], lmbda_grad[r,0], c_grad[r,0], phi_grad[r,0], logsigmasq_grad[r,0] = results1[r][1][:]
        alpha_grad[r,1], lmbda_grad[r,1], c_grad[r,1], phi_grad[r,1], logsigmasq_grad[r,1] = results2[r][1][:]
        logNC[r,0], logNC[r,1] = results1[r][0], results2[r][0]
    results1, results2 = [], []
    gc.collect()
    
    return alpha_grad, lmbda_grad, c_grad, phi_grad, logsigmasq_grad, logNC
    
    

#####################################################################################################################
############################################## NON-REVERSIBLE SAMPLERS ##############################################
#####################################################################################################################

def alpha_func(zeta, deltat) :
    return np.exp(-zeta*deltat)

def G(sigma, zeta, deltat) :
    if zeta == 0 :
        return sigma*np.sqrt(deltat)
    else :
        return sigma*np.sqrt((1-np.exp(-2*deltat*zeta))/(2*zeta))
    
    

def ODABADO(Y, x_0, n_particles, theta, N, M, deltat, nu, p, zeta, beta=1, sigmaA=0) :
    
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    J, K = np.shape(lmbda)
    I, K = np.shape(x_0)
    
    M_alpha, M_lmbda = M[:]
    nu_alpha, nu_lmbda = nu[:]
    p_alpha, p_lmbda = p[:]
    zeta_alpha, zeta_lmbda = zeta[:]
    Minv_alpha = np.linalg.inv(M_alpha)
    Minv_lmbda = np.linalg.inv(M_lmbda)
    
    zeta_alpha_vals = np.ones(N+1)
    zeta_alpha_vals[0] = zeta_alpha
    zeta_lmbda_vals = np.ones(N+1)
    zeta_lmbda_vals[0] = zeta_lmbda

    alpha_chain = np.zeros((N+1,J))
    alpha_chain[0] = alpha
    lmbda_chain = np.zeros((N+1,J*K))
    lmbda_chain[0] = lmbda.flatten()
    
    for n in trange(N) :
        zeta_alpha = zeta_alpha_vals[n]
        zeta_lmbda = zeta_lmbda_vals[n]
        q_alpha = np.copy(alpha_chain[n])
        q_lmbda = np.copy(lmbda_chain[n])

        p_alpha = alpha_func(zeta_alpha, deltat/2)*p_alpha + G(sigmaA, zeta_alpha, deltat/2)*npr.randn(J)
        zeta_alpha = zeta_alpha + deltat/(2*nu_alpha)*(np.linalg.norm(p_alpha)**2 - J/beta)
        q_alpha = q_alpha + deltat/2*np.matmul(Minv_alpha, p_alpha)

        p_lmbda = alpha_func(zeta_lmbda, deltat/2)*p_lmbda + G(sigmaA, zeta_lmbda, deltat/2)*npr.randn(J*K)
        zeta_lmbda = zeta_lmbda + deltat/(2*nu_lmbda)*(np.linalg.norm(p_lmbda)**2 - J*K/beta)
        q_lmbda = q_lmbda + deltat/2*np.matmul(Minv_lmbda, p_lmbda)

        theta = [q_alpha, np.reshape(q_lmbda, (J,K)), c, phi, logsigmasq] 
        _, grad_est = block_PF(Y, x_0, n_particles, theta, calc_grad=True)
        grad_alpha, grad_lmbda = grad_est[0], (grad_est[1]).flatten()

        phat_alpha = p_alpha - deltat*grad_alpha
        q_alpha = q_alpha + deltat/2*np.matmul(Minv_alpha, phat_alpha)
        zeta_alpha = zeta_alpha + deltat/(2*nu_alpha)*(np.linalg.norm(phat_alpha)**2 - J/beta)
        p_alpha = alpha_func(zeta_alpha, deltat/2)*phat_alpha + G(sigmaA, zeta_alpha, deltat/2)*npr.randn(J)

        phat_lmbda = p_lmbda - deltat*grad_lmbda
        q_lmbda = q_lmbda + deltat/2*np.matmul(Minv_lmbda, phat_lmbda)
        zeta_lmbda = zeta_lmbda + deltat/(2*nu_lmbda)*(np.linalg.norm(phat_lmbda)**2 - J*K/beta)
        p_lmbda = alpha_func(zeta_lmbda, deltat/2)*phat_lmbda + G(sigmaA, zeta_lmbda, deltat/2)*npr.randn(J*K)

        alpha_chain[n+1] = q_alpha
        zeta_alpha_vals[n+1] = zeta_alpha
        lmbda_chain[n+1] = q_lmbda
        zeta_lmbda_vals[n+1] = zeta_lmbda
        
    return [alpha_chain, lmbda_chain], [zeta_alpha_vals, zeta_lmbda_vals]
    
    
    
def get_corrs(lmbda_chain) :
    N, J, K = np.shape(lmbda_chain)
    corrs = np.zeros((N,J,J))
    for j1 in range(J) :
        for j2 in range(J) :
            corrs[:,j1,j2] = np.sum(lmbda_chain[:,j1]*lmbda_chain[:,j2],axis=1)\
                                    /(np.linalg.norm(lmbda_chain[:,j1],axis=1)*np.linalg.norm(lmbda_chain[:,j2],axis=1))
    return corrs

def plot_non_reversible(alpha_chain, lmbda_chain, burnin=100) :
    N, J, K = np.shape(lmbda_chain)
    plt.rcParams['figure.figsize'] = (12, 3)
    
    plt.subplot(121)
    plt.plot(alpha_chain[burnin:])
    #for j in range(J) :
        #plt.axhline(y=alpha[j], color="k")
    plt.grid(True)
    plt.title(r"$\alpha$", fontsize=15)
    
    plt.subplot(122)
    lmbda_chain = np.reshape(lmbda_chain, (N,J*K))
    plt.plot(lmbda_chain[burnin:])
    #for idx in range(J*K) :
    #    plt.axhline(y=(lmbda.flatten())[idx], color="k")
    plt.grid(True)
    plt.title(r"$\lambda$", fontsize=15)
    plt.show()
    
    
    
