from __future__ import division
import numpy as np, time, matplotlib.pyplot as plt, math, pandas, numpy.random as npr, multiprocessing as mp, torch, copy
from time import time
from scipy.stats import *
from tqdm import trange
import scipy


def propagate(particles, theta) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    sigma = torch.exp(logsigmasq/2)
    return c + phi*particles + sigma*torch.randn(*np.shape(particles)) 

def simulate_data(x_0, T, J, theta) :   # I = no. of locations, J = no. of species, K = no. of latent factors
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    I, K = np.shape(x_0)
    Y = torch.zeros(T,I,J, requires_grad=False)
    X = torch.zeros(T+1,I,K, requires_grad=False)
    X[0] = x_0

    for t in range(T) :
        X[t+1] = propagate(X[t], theta)
        reg = alpha + torch.matmul(lmbda,X[t+1].transpose(0,1))
        probs = 1/(1+np.exp(-reg.detach().numpy()))
        Y[t,:] = torch.tensor(npr.binomial(n=1, p=probs).transpose())
    return Y, X

#####################################################################################################################
#####################################      Bootstrap particle filter     ############################################
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
    n_particles = np.shape(particles)[-1]
    I, J = np.shape(y)
    prob = (1/(1+torch.exp(-torch.add(alpha, torch.matmul(lmbda,particles))))).transpose(0,2)
    return torch.tensor(np.prod((prob**(y.transpose(0,1))*(1-prob)**(1-y.transpose(0,1))).detach().numpy(), (1,2)))

# --------------------------------------------- Track particles ------------------------------------------------- #

def bootstrap_pf_track(Y, x_0, n_particles, theta) :
    
    np.random.seed()
    scipy.random.seed()
    
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

# ----------------------------- Functions to calculate gradient using Fisher's identity ----------------------------- #
################################### USE AUTOMATIC DIFFERENTIATION ##########################################

def update_grad(y, T, theta, propagated_particles, particles) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    b = torch.add(alpha, torch.matmul(lmbda,particles))
    ll1 = -1/(2*torch.exp(logsigmasq))*torch.sum(torch.sum((propagated_particles - c - phi*particles)**2, [0,1]))
    ll2 = torch.sum(torch.sum(b.transpose(0,2)*((y-1).transpose(0,1)), [1,2]))
    ll3 = -(1/2)*(torch.sum(alpha**2) + torch.sum(lmbda**2) + torch.sum(c**2) + torch.sum(phi**2) + torch.sum(logsigmasq**2))/T
    ll = ll1 + ll2 + ll3
    ll.backward(retain_graph=True)
    
def weigh_grad(theta, weights) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]


def bootstrap_PF_grad_autodiff(x_0, n_particles, theta, Y) :
    
    np.random.seed()
    scipy.random.seed()
    
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    T, I, J = np.shape(Y)
    I, K = np.shape(x_0)
    particles, weights, logNC = torch.zeros((I,K,n_particles)), torch.ones(n_particles, requires_grad=False)/n_particles, torch.tensor(0.)
    for n in range(n_particles) :
        particles[:,:,n] = x_0
    for t in range(T) :
        resampled_idx = npr.choice(a=n_particles, size=n_particles, p=weights.numpy())
        propagated_particles = propagate(particles[:,:,resampled_idx], theta)
        incremental_weights = potential(Y[t], propagated_particles, theta)
        weights = weights*incremental_weights
        logNC = logNC + torch.log(torch.sum(weights))
        weights = weights/torch.sum(weights) 
        update_grad(Y[t], T, theta, propagated_particles, particles[:,:,resampled_idx])
        particles = propagated_particles.clone()
    return logNC


#####################################################################################################################
############################################### Pseudo-marginal MALA ################################################
#####################################################################################################################

############################################## Without using autograd ###############################################

def propose_mala(theta, grad, tau) :
    np.random.seed()
    scipy.random.seed()
    
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    grad_alpha, grad_lmbda, grad_c, grad_phi, grad_logsigmasq = grad[:]
    
    alpha_proposed = alpha + tau*grad_alpha + np.sqrt(2*tau)*npr.randn(*np.shape(alpha))
    lmbda_proposed = lmbda + tau*grad_lmbda + np.sqrt(2*tau)*npr.randn(*np.shape(lmbda))
    c_proposed = c + tau*grad_c + np.sqrt(2*tau)*npr.randn()
    phi_proposed = phi + tau*grad_phi + np.sqrt(2*tau)*npr.randn()
    logsigmasq_proposed = logsigmasq + tau*grad_logsigmasq + np.sqrt(2*tau)*npr.randn()

def transition_prob_mala(theta_current, theta_proposed, ll_current, ll_proposed, grad_current, grad_proposed, tau, power) :
    log_prior_current, log_prior_proposed = log_prior(theta_current), log_prior(theta_proposed) 
    a = power*(ll_proposed-ll_current) + (log_prior_proposed-log_prior_current)
    b1 = -np.sum([np.linalg.norm(theta_current[i] - theta_proposed[i] - tau*grad_proposed[i])**2 for i in range(5)])/(4*tau)
    b2 = -np.sum([np.linalg.norm(theta_proposed[i] - theta_current[i] - tau*grad_current[i])**2 for i in range(5)])/(4*tau)
    return a + b1 - b2

def pMCMC_mala(x_0, Y, theta_0, n_particles, n_mcmc, tau=1e-3, power=1) :
    
    np.random.seed()
    scipy.random.seed()
    
    alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain, lls, theta_mu, theta_m2 = initialise(theta_0, n_mcmc)
    theta_chain = [alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain]
    
    theta_current = [alpha_chain[0], lmbda_chain[0], c_chain[0], phi_chain[0], logsigmasq_chain[0]]    
    ll_current, grad_current = bootstrap_PF_grad(x_0, n_particles, theta_current, Y)
    accepted = 0
    last_jump = 0
    
    accept_probs = np.zeros(n_mcmc)
    
    for n in trange(n_mcmc) :
        theta_proposed = propose_mala(theta_current, grad_current, tau)
        ll_proposed, grad_proposed = bootstrap_PF_grad(x_0, n_particles, theta_proposed, Y)
        
        log_accept_prob = transition_prob_mala(theta_current, theta_proposed, ll_current, ll_proposed, grad_current, grad_proposed, tau, power) 
        accept_probs[n] = np.exp(log_accept_prob)
        
        if np.log(npr.rand()) < log_accept_prob :
            ll_current = np.copy(ll_proposed)
            theta_current = np.copy(theta_proposed)
            grad_current = np.copy(grad_proposed)
            accepted += 1
            last_jump = n
        else :
            if n - last_jump > 50 :
                ll_current, grad_current = bootstrap_PF_grad(x_0, n_particles, theta_current, Y)
        
        theta_chain = push(theta_chain, theta_current, n+1)

    print(100*accepted/n_mcmc, "% acceptance rate")
    return theta_chain, accept_probs



############################################## Attempt using autograd ###############################################

def propose_mala_autograd(theta, tau) :
    
    np.random.seed()
    scipy.random.seed()
    
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    alpha_proposed = torch.clone(alpha.detach() + tau[0]*alpha.grad.detach() + np.sqrt(2*tau[0])*torch.randn(*np.shape(alpha)).detach())
    lmbda_proposed = torch.clone(lmbda.detach() + tau[1]*lmbda.grad.detach() + np.sqrt(2*tau[1])*torch.randn(*np.shape(lmbda)).detach())
    c_proposed = torch.clone(c.detach() + tau[2]*c.grad.detach() + np.sqrt(2*tau[2])*torch.randn(1).detach())
    phi_proposed = torch.clone(phi.detach() + tau[3]*phi.grad.detach() + np.sqrt(2*tau[3])*torch.randn(1).detach())
    logsigmasq_proposed = torch.clone(logsigmasq.detach() + tau[4]*logsigmasq.grad.detach() + np.sqrt(2*tau[4])*torch.randn(1).detach())
    
    return [alpha_proposed.requires_grad_(True), lmbda_proposed.requires_grad_(True), c_proposed.requires_grad_(True), \
            phi_proposed.requires_grad_(True), logsigmasq_proposed.requires_grad_(True)]

def transition_prob_mala_autograd(theta_current, theta_proposed, ll_current, ll_proposed, tau, power) :
    log_prior_current, log_prior_proposed = log_prior(theta_current), log_prior(theta_proposed) 
    a = power*(ll_proposed-torch.tensor(ll_current)) + (log_prior_proposed-log_prior_current)
    b1 = -sum([torch.norm(theta_current[i] - theta_proposed[i] - tau[i]*theta_proposed[i].grad).detach()**2/(4*tau[i]) for i in range(5)])
    b2 = -sum([torch.norm(theta_proposed[i] - theta_current[i] - tau[i]*theta_current[i].grad).detach()**2/(4*tau[i]) for i in range(5)])
    return a + b1 - b2

def zero_grad(theta) :
    alpha, lmbda, c, phi, logsigmasq = theta[:]
    alpha.grad = torch.zeros(*np.shape(alpha))
    lmbda.grad = torch.zeros(*np.shape(lmbda))
    c.grad = torch.tensor(0.)
    phi.grad = torch.tensor(0.)
    logsigmasq.grad = torch.tensor(0.)

# THIS IS WRONG
def pMCMC_mala_autograd(x_0, Y, theta_0, n_particles, n_mcmc, tau, power=1) :
    
    np.random.seed()
    scipy.random.seed()
    
    alpha, lmbda, c, phi, logsigmasq = theta_0[:]
    alpha_chain = torch.zeros((n_mcmc+1,*np.shape(alpha)), requires_grad=False)
    lmbda_chain = torch.zeros((n_mcmc+1,*np.shape(lmbda)), requires_grad=False)
    c_chain = torch.zeros(n_mcmc+1, requires_grad=False)
    phi_chain = torch.zeros(n_mcmc+1, requires_grad=False)
    logsigmasq_chain = torch.zeros(n_mcmc+1, requires_grad=False)
    
    theta_chain = [alpha_chain, lmbda_chain, c_chain, phi_chain, logsigmasq_chain]
    
    theta_current = theta_0 
    ll_current = bootstrap_PF_grad_autodiff(x_0, n_particles, theta_current, Y)
    accepted = 0
    last_jump = 0
    accept_probs = np.zeros(n_mcmc)
    
    for n in trange(n_mcmc) :
        theta_proposed = propose_mala_autograd(theta_current, tau)
        ll_proposed = bootstrap_PF_grad_autodiff(x_0, n_particles, theta_proposed, Y)
        
        log_accept_prob = transition_prob_mala_autograd(theta_current, theta_proposed, ll_current, ll_proposed, tau, power) 
        accept_probs[n] = np.exp(log_accept_prob)
        
        if np.log(npr.rand()) < log_accept_prob :
            ll_current = np.copy(ll_proposed)
            theta_current = theta_proposed
            accepted += 1
            last_jump = n
        else :
            if n - last_jump > 50 :
                # set theta gradient to 0
                zero_grad(theta_current)
                ll_current = bootstrap_PF_grad_autodiff(x_0, n_particles, theta_current, Y)
        
        theta_chain = push(theta_chain, theta_current, n+1)

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

