from __future__ import division
import numpy as np, time, matplotlib.pyplot as plt, math, pandas, numpy.random as npr, multiprocessing as mp
from time import time
from scipy.stats import *
from tqdm import trange


def propagate(particles, theta) :
    c, phi, logsigmasq = theta[2], theta[3], theta[4]
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
            #Y[t,:,j] = (reg + npr.randn(*np.shape(reg))).transpose()
    
    return Y, X

#####################################################################################################################
#####################################      Bootstrap particle filter     ############################################
#####################################################################################################################

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
    alpha, lmbda = theta[0], theta[1]
    n_particles = np.shape(particles)[-1]
    I, J = np.shape(y)
    reg = np.zeros((J,n_particles,I))
    for j in range(J) :
        reg[j] = alpha[j] + lmbda[j].dot(particles.transpose())
    prob = (1/(1+np.exp(-reg))).transpose()
    return np.asarray([np.prod((prob[:,i,:]**y)*(1-prob[:,i,:])**(1-y)) for i in range(n_particles)])


def bootstrap_PF(x_0, n_particles, theta, Y) :
    T, I, J = np.shape(Y)
    particles, weights, logNC = np.zeros((*np.shape(x_0),n_particles)), np.ones(n_particles)/n_particles, 0.
    for n in range(n_particles) :
        particles[:,:,n] = x_0
    for t in range(T) :
        particles = propagate(particles, theta)
        incremental_weights = potential(Y[t], particles, theta)
        weights = weights*incremental_weights
        logNC += np.log(np.sum(weights))
        weights, particles = adaptive_resample(weights, particles)                
    return logNC

def bootstrap_pf_track(Y, x_0, n_particles, theta) :
    
    T, I, J = np.shape(Y)
    K = np.shape(x_0)[-1]
    particles = np.zeros((T+1,*np.shape(x_0),n_particles))   # I = number of locations, K = dimension at each location
    for n in range(n_particles) :
        particles[0,:,:,n] = x_0
    weights = np.ones(n_particles)/n_particles 
    logNC = 0
    
    for t in trange(T) :
        particles[t+1] = propagate(particles[t], theta)
        incremental_weights = potential(Y[t], particles[t+1], theta)
        weights = weights*incremental_weights
        logNC += np.log(np.sum(weights))
        resampled_idx = npr.choice(a=n_particles, size=n_particles, p=weights/np.sum(weights))
        part = particles[t+1]
        particles[t+1] = part[:,:,resampled_idx]
        weights = np.ones(n_particles)/n_particles
    return particles, logNC

#####################################################################################################################
#######################################      Block particle filter     ##############################################
#####################################################################################################################


def local_potential(y, particles, theta) :
    alpha, lmbda = theta[0], theta[1]
    n_particles = np.shape(particles)[0]
    
    J = len(y)
    reg = np.zeros((J,n_particles))
    for j in range(J) :
        reg[j] = alpha[j] + lmbda[j].dot(particles.transpose())
    prob = (1/(1+np.exp(-reg)))
    return np.asarray([np.prod((prob[:,n]**y)*(1-prob[:,n])**(1-y)) for n in range(n_particles)])

# this can be speeded up..
def block_pf(Y, x_0, n_particles, theta) : # I = number of locations, K = dimension at each location
    
    T, I, J = np.shape(Y)
    K = np.shape(x_0)[-1]
    particles = np.zeros((T+1,n_particles,I,K))  
    particles[0] = x_0
    weights = np.ones((n_particles,I))/n_particles 
    
    for t in trange(T) :
        # propagation:
        particles[t+1] = propagate(particles[t], theta)  

        #weighting:
        for i in range(I) :
            weights[:,i] = local_potential(Y[t,i], particles[t+1,:,i], theta)
            
        # resampling:
        for i in range(I) :
            resampled_idx = npr.choice(a=n_particles, size=n_particles, p=weights[:,i]/np.sum(weights[:,i]))
            particles[t+1,:,i] = particles[t+1,resampled_idx,i]
            
    return particles, weights

#####################################################################################################################
#####################################      Pseudo-marginal MCMC stuff     ###########################################
#####################################################################################################################

def initialise(theta_0, n_mcmc) :
    alpha, lmbda, c, phi, logsigmasq = theta_0[:]
    
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

def propose(theta, scale) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    scale_alpha, scale_lmbda, scale_c, scale_phi, scale_logsigmasq = scale[:]
    
    alpha_proposed = npr.multivariate_normal(alpha, scale_alpha)
    #alpha_proposed = alpha + scale_alpha*npr.randn(*np.shape(alpha))
    lmbda_proposed = npr.multivariate_normal(lmbda.reshape(np.prod(np.shape(lmbda))), scale_lmbda).reshape(*np.shape(lmbda))
    #lmbda_proposed = lmbda + scale_lmbda*npr.randn(*np.shape(lmbda))
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
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    return np.sum(-1/2*(sum(alpha**2) + sum(lmbda**2) + c**2 + phi**2 + logsigmasq**2))

def pMCMC(x_0, Y, theta_0, n_particles, n_mcmc, scale, power=1, adapt=True, start_adapt=0.2) :
    alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain, lls, theta_mu, theta_m2 = initialise(theta_0, n_mcmc)
    theta_chain = [alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain]
    
    theta_current = [alpha_chain[0], lmbda_chain[0], c_chain[0], phi_chain[0], logsigmasq_chain[0]]    
    lls[0] = bootstrap_PF(x_0, n_particles, theta_current, Y)
    accepted = 0
    last_jump = 0
    
    accept_probs = np.zeros(n_mcmc)
    
    for n in trange(n_mcmc) :
        theta_proposed = propose(theta_current, scale)
        ll_proposed = bootstrap_PF(x_0, n_particles, theta_proposed, Y)
        log_prior_current, log_prior_proposed = log_prior(theta_current), log_prior(theta_proposed) 
        log_accept_prob = power*(ll_proposed-lls[n]) + (log_prior_proposed-log_prior_current)
        accept_probs[n] = np.exp(log_accept_prob)
        
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
    return theta_chain, scale, accept_probs

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

