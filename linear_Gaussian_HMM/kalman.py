from __future__ import division
from pykalman import KalmanFilter
import numpy as np, numpy.random as npr, copy, matplotlib.pyplot as plt, multiprocessing as mp, scipy, gc
from scipy.stats import *
from tqdm import trange
from pylab import plot, show, legend

def generate_data_linear_gaussian(mu0, Sigma0, A, C, Q, R, T) :
    npr.seed()
    scipy.random.seed()
    
    dim = np.shape(A)[0]
    Y = np.zeros((T,dim))
    X = np.zeros((T,dim))
    X[0] = multivariate_normal.rvs(mean=mu0, cov=Sigma0)
    for t in np.arange(1,T) :
        X[t] = multivariate_normal.rvs(mean=np.matmul(A,X[t-1]), cov=Q)
    for t in range(T) :
        Y[t] = multivariate_normal.rvs(mean=np.matmul(C,X[t]), cov=R)
    return Y, X


def predictive(Y, mu0, Sigma0, A, C, Q, R) :
    
    npr.seed()
    scipy.random.seed()
    
    kf = KalmanFilter(initial_state_mean=mu0,
                      initial_state_covariance=Sigma0,
                      transition_matrices=A, 
                      observation_matrices=C, 
                      transition_covariance=Q, 
                      observation_covariance=R)
    (filtered_state_means, filtered_state_covariances) = kf.filter(Y)
    
    predicted_state_means = np.zeros((np.shape(filtered_state_means)))
    predicted_state_covariances = np.zeros(np.shape(filtered_state_covariances))
    
    T = np.shape(Y)[0]
    for t in np.arange(1,T) :
        predicted_state_means[t] = np.matmul(A,filtered_state_means[t-1])
        predicted_state_covariances[t] = np.matmul(A,np.matmul(filtered_state_covariances[t-1],A.transpose())) + Q
        
    return predicted_state_means, predicted_state_covariances

#################################### LOG-LIKELIHOOD USING KALMAN FILTER ####################################

def log_likelihood(Y, A, C, sigmax2, sigmay2, mu0, Sigma0) :
    dim = np.shape(A)[0]
    T = np.shape(Y)[0]
    Q = sigmax2*np.eye(dim)
    R = sigmay2*np.eye(dim)
    
    predicted_state_means, predicted_state_covariances = predictive(Y, mu0, Sigma0, A, C, Q, R)
    
    lpdf = multivariate_normal.logpdf(x=Y[0], 
                                      mean=np.matmul(C,mu0), 
                                      cov=np.matmul(np.matmul(C,Sigma0),C.transpose())+R)
    for t in np.arange(1,T) :
        lpdf += multivariate_normal.logpdf(x=Y[t], 
                                           mean=np.matmul(C,predicted_state_means[t]), 
                                           cov=np.matmul(np.matmul(C,predicted_state_covariances[t]),C.transpose())+R)
    return lpdf


################################# LOG-LIKELIHOOD USING BOOTSTRAP PARTICLE FILTER #################################

def propagate(particles, A, Q) :
    npr.seed()
    scipy.random.seed()
    n_particles, dim = np.shape(particles)
    return (A.dot(particles.transpose())).transpose() + multivariate_normal.rvs(mean=np.zeros(dim), cov=Q, size=n_particles)

def resample(particles, weights) :
    npr.seed()
    scipy.random.seed()
    n_particles = len(weights)
    particles = particles[npr.choice(a=n_particles,size=n_particles,p=weights/np.sum(weights))]
    return particles, np.ones(n_particles)/n_particles


def potential(particles, y, C, R) :
    dim = len(y)
    mean = C.dot(particles.transpose()).transpose()
    Rinv = np.linalg.inv(R)
    log_wts = -(1/2)*(np.diag(((y-mean).dot(Rinv).dot((y-mean).transpose()))) \
                      + np.log(np.linalg.det(R)) + dim*np.log(2*np.pi))
    return np.exp(log_wts)

def bootstrap_PF(Y, A, C, sigmax2, sigmay2, mu0, Sigma0, n_particles) :
    
    npr.seed()
    scipy.random.seed()
    T, dim = np.shape(Y)
    Q = sigmax2*np.eye(dim)
    R = sigmay2*np.eye(dim)
    particles = np.zeros((T,n_particles,dim))
    
    particles[0] = multivariate_normal.rvs(mean=mu0, cov=Sigma0, size=n_particles)
    weights = potential(particles[0], Y[0], C, R)/n_particles
    logNC = np.log(np.sum(weights))
    particles[0], weights = resample(particles[0], weights)
    
    for t in np.arange(1,T) :
        particles[t] = propagate(particles[t-1], A, Q)
        weights = weights*potential(particles[t], Y[t], C, R)
        logNC += np.log(np.sum(weights))
        particles[t], weights = resample(particles[t], weights)
                
    return logNC, particles

def bootstrap_PF_mult(Y, A, C, sigmax2, sigmay2, mu0, Sigma0, n_particles, m) :
    npr.seed()
    scipy.random.seed()
    global f
    def f(n_particles) :
        return bootstrap_PF(Y, A, C, sigmax2, sigmay2, mu0, Sigma0, n_particles)[0]
    pool = mp.Pool(min(m,10))
    result = pool.map(f, [n_particles for n_particles in [n_particles]*m])
    pool.close()
    gc.collect()
    return np.asarray(result)

#################################### METROPOLIS-HASTINGS ALGORITHM ####################################


def adaptive_MH(Y, A, C, sigmax2, sigmay2, mu0, Sigma0, n_mcmc, scale, 
                method="kalman", n_particles=100, adapt=True, start_adapt=0.2, power=1) :
    
    npr.seed()
    scipy.random.seed()
    
    log_sigmax2_chain = np.zeros(n_mcmc+1)
    log_sigmax2_chain[0] = np.log(sigmax2)
    log_sigmay2_chain = np.zeros(n_mcmc+1)
    log_sigmay2_chain[0] = np.log(sigmay2)
    accepted = 0
    last_accepted = 0
    
    scales = np.zeros((n_mcmc+1,2))
    scales[:] = scale
    
    log_sigmax2_mu = log_sigmax2_chain[0]
    log_sigmax2_m2 = log_sigmax2_chain[0]**2
    log_sigmay2_mu = log_sigmay2_chain[0]
    log_sigmay2_m2 = log_sigmay2_chain[0]**2
    
    lls = np.zeros((n_mcmc+1,power)) 
    
    if method == "kalman" :
        lls[0] = log_likelihood(Y, A, C, sigmax2, sigmay2, mu0, Sigma0)
    elif method == "particle" :
        lls[0] = bootstrap_PF_mult(Y, A, C, sigmax2, sigmay2, mu0, Sigma0, n_particles, power)
    
    for n in trange(n_mcmc) :
        log_sigmax2_proposed = log_sigmax2_chain[n] + scales[n,0]*npr.randn()
        log_sigmay2_proposed = log_sigmay2_chain[n] + scales[n,1]*npr.randn()
        
        if method == "kalman" :
            ll_proposed = log_likelihood(Y, A, C, np.exp(log_sigmax2_proposed), np.exp(log_sigmay2_proposed), mu0, Sigma0)
            log_accept_ratio = power*(ll_proposed - ll_current)
        elif method == "particle" :
            ll_proposed = bootstrap_PF_mult(Y, A, C, np.exp(log_sigmax2_proposed), np.exp(log_sigmay2_proposed), 
                                            mu0, Sigma0, n_particles, power)
            log_accept_ratio = np.sum(ll_proposed) - np.sum(lls[n])
        
        log_accept_ratio += norm.logpdf(x=log_sigmax2_proposed, loc=0, scale=10) - \
                            norm.logpdf(x=log_sigmax2_chain[n], loc=0, scale=10) + \
                            norm.logpdf(x=log_sigmay2_proposed, loc=0, scale=10) - \
                            norm.logpdf(x=log_sigmay2_chain[n], loc=0, scale=10) + \
                            (log_sigmax2_proposed - log_sigmax2_chain[n]) + (log_sigmay2_proposed - log_sigmay2_chain[n])
        
        if np.log(npr.rand()) < log_accept_ratio :
            lls[n+1] = ll_proposed
            log_sigmax2_chain[n+1] = log_sigmax2_proposed
            log_sigmay2_chain[n+1] = log_sigmay2_proposed
            accepted += 1
            last_accepted = n
        else :
            log_sigmax2_chain[n+1] = log_sigmax2_chain[n]
            log_sigmay2_chain[n+1] = log_sigmay2_chain[n]
            lls[n+1] = lls[n]
        
        if (method == "particle") & (n - last_accepted > 50) :
            lls[n+1] = bootstrap_PF_mult(Y, A, C, np.exp(log_sigmax2_chain[n+1]), np.exp(log_sigmay2_chain[n+1]), 
                                           mu0, Sigma0, n_particles, power)
            
        log_sigmax2_mu = ((n+1)*log_sigmax2_mu + log_sigmax2_chain[n+1])/(n+2)    
        log_sigmax2_m2 = ((n+1)*log_sigmax2_m2 + log_sigmax2_chain[n+1]**2)/(n+2)
        log_sigmay2_mu = ((n+1)*log_sigmay2_mu + log_sigmay2_chain[n+1])/(n+2)    
        log_sigmay2_m2 = ((n+1)*log_sigmay2_m2 + log_sigmay2_chain[n+1]**2)/(n+2)
        
        if adapt :
            if n >= int(n_mcmc*start_adapt) : 
                scales[n+1,0] = np.sqrt((log_sigmax2_m2 - log_sigmax2_mu**2))
                scales[n+1,1] = np.sqrt((log_sigmay2_m2 - log_sigmay2_mu**2))
    
    print(100*accepted/n_mcmc, "% acceptance rate")
    return log_sigmay2_chain, log_sigmax2_chain, accepted, scales


############################################ PLOT ##############################################

def plot(log_sigmax2_chain, log_sigmay2_chain, sigmax2, sigmay2, title) :
    n_mcmc = len(log_sigmax2_chain)
    plt.rcParams['figure.figsize'] = (10, 3)
    plt.subplot(121)
    plt.plot(log_sigmax2_chain)
    plt.axhline(y=np.log(sigmax2), color="red")
    plt.title(r"$\log \, \sigma_x^2$", fontsize=14)
    plt.xlabel("MCMC iteration")
    plt.grid(True)
    plt.subplot(122)
    plt.plot(log_sigmay2_chain)
    plt.axhline(y=np.log(sigmay2), color="red")
    plt.title(r"$\log \, \sigma_y^2$", fontsize=14)
    plt.xlabel("MCMC iteration")
    plt.grid(True)
    plt.suptitle(title);



