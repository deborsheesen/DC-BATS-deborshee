from __future__ import division
from pykalman import KalmanFilter
import numpy as np, numpy.random as npr, copy, matplotlib.pyplot as plt, multiprocessing as mp, scipy, gc
from scipy.stats import *
from tqdm import trange
from pylab import plot, show, legend
from time import time
import copy


def predictive(Y, Z, mu0, Sigma0, A, C, B, Q, R) :
    
    b = np.transpose(np.matmul(B,Z.transpose()))
    kf = KalmanFilter(initial_state_mean=mu0,
                      initial_state_covariance=Sigma0,
                      transition_matrices=A, 
                      observation_matrices=C, 
                      observation_offsets=b,
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

def log_likelihood(Y, Z, A, C, B, sigmax2, sigmay2, mu0, Sigma0) :
    lat_dim = np.shape(A)[0]
    obs_dim = np.shape(C)[0]
    T = np.shape(Y)[0]
    Q = sigmax2*np.eye(lat_dim)
    R = sigmay2*np.eye(obs_dim)
    b = np.transpose(np.matmul(B,Z.transpose()))
    predicted_state_means, predicted_state_covariances = predictive(Y, Z, mu0, Sigma0, A, C, B, Q, R)
    
    lpdf = multivariate_normal.logpdf(x=Y[0], 
                                      mean=np.matmul(C,mu0)+b[0], 
                                      cov=np.matmul(np.matmul(C,Sigma0),C.transpose())+R)
    for t in np.arange(1,T) :
        lpdf += multivariate_normal.logpdf(x=Y[t], 
                                           mean=np.matmul(C,predicted_state_means[t])+b[t], 
                                           cov=np.matmul(np.matmul(C,predicted_state_covariances[t]),C.transpose())+R)
    return lpdf


#################################### METROPOLIS-HASTINGS ALGORITHM ####################################


def adaptive_MH(Y, Z, A, C, B, sigmax2, sigmay2, mu0, Sigma0, n_mcmc, scale, scale_A, scale_C, scale_B,
                adapt=True, start_adapt=0.2, power=1, kappa=1) :
    
    npr.seed()
    scipy.random.seed()
    
    log_sigmax2_chain = np.zeros(n_mcmc+1)
    log_sigmax2_chain[0] = np.log(sigmax2)
    log_sigmay2_chain = np.zeros(n_mcmc+1)
    log_sigmay2_chain[0] = np.log(sigmay2)
    A_chain = np.zeros((n_mcmc+1, *np.shape(A)))
    C_chain = np.zeros((n_mcmc+1, *np.shape(C)))
    B_chain = np.zeros((n_mcmc+1, *np.shape(B)))
    A_chain[0], C_chain[0], B_chain[0] = A, C, B
    
    scales = np.zeros((n_mcmc+1,2))
    scales[:] = scale
    scales_A = np.zeros((n_mcmc+1,*np.shape(A)))
    scales_C = np.zeros((n_mcmc+1,*np.shape(C)))
    scales_B = np.zeros((n_mcmc+1,*np.shape(B)))
    scales_A[:], scales_C[:], scales_B[:] = scale_A, scale_C, scale_B
    
    log_sigmax2_mu = log_sigmax2_chain[0]
    log_sigmax2_m2 = log_sigmax2_chain[0]**2
    log_sigmay2_mu = log_sigmay2_chain[0]
    log_sigmay2_m2 = log_sigmay2_chain[0]**2
    A_mu, C_mu, B_mu = A_chain[0], C_chain[0], B_chain[0]
    A_m2, C_m2, B_m2 = A_chain[0]**2, C_chain[0]**2, B_chain[0]**2
    
    accepted = 0
    last_accepted = 0
    lls = np.zeros(n_mcmc+1)
    
    start = time()
    lls[0] = log_likelihood(Y, Z, A, C, B, sigmax2, sigmay2, mu0, Sigma0)
    
    for n in range(n_mcmc) :
        log_sigmax2_proposed = log_sigmax2_chain[n] + scales[n,0]*npr.randn()
        log_sigmay2_proposed = log_sigmay2_chain[n] + scales[n,1]*npr.randn()
        A_proposed = A_chain[n] + scales_A[n]*npr.randn(*np.shape(A))
        C_proposed = C_chain[n] + scales_C[n]*npr.randn(*np.shape(C))
        B_proposed = B_chain[n] + scales_B[n]*npr.randn(*np.shape(B))
        
        ll_proposed = log_likelihood(Y, Z, A_proposed, C_proposed, B_proposed,
                                     np.exp(log_sigmax2_proposed), np.exp(log_sigmay2_proposed), mu0, Sigma0)
        
        log_accept_ratio = power*(ll_proposed - lls[n])
        log_accept_ratio += norm.logpdf(x=log_sigmax2_proposed, loc=0, scale=10) - \
                            norm.logpdf(x=log_sigmax2_chain[n], loc=0, scale=10) + \
                            norm.logpdf(x=log_sigmay2_proposed, loc=0, scale=10) - \
                            norm.logpdf(x=log_sigmay2_chain[n], loc=0, scale=10) + \
                            (log_sigmax2_proposed - log_sigmax2_chain[n]) + (log_sigmay2_proposed - log_sigmay2_chain[n]) + \
                            np.sum(norm.logpdf(x=A_proposed, loc=np.eye(*np.shape(A)), scale=10)) - \
                            np.sum(norm.logpdf(x=A_chain[n], loc=np.eye(*np.shape(A)), scale=10)) + \
                            np.sum(norm.logpdf(x=C_proposed, loc=np.eye(*np.shape(C)), scale=10)) - \
                            np.sum(norm.logpdf(x=C_chain[n], loc=np.eye(*np.shape(C)), scale=10)) + \
                            np.sum(norm.logpdf(x=B_proposed, loc=np.eye(*np.shape(B)), scale=10)) - \
                            np.sum(norm.logpdf(x=B_chain[n], loc=np.eye(*np.shape(B)), scale=10))
        
        if np.log(npr.rand()) < log_accept_ratio :
            lls[n+1] = ll_proposed
            log_sigmax2_chain[n+1] = log_sigmax2_proposed
            log_sigmay2_chain[n+1] = log_sigmay2_proposed
            A_chain[n+1] = A_proposed
            C_chain[n+1] = C_proposed
            B_chain[n+1] = B_proposed
            accepted += 1
            last_accepted = n
        else :
            log_sigmax2_chain[n+1] = log_sigmax2_chain[n]
            log_sigmay2_chain[n+1] = log_sigmay2_chain[n]
            A_chain[n+1] = A_chain[n]
            C_chain[n+1] = C_chain[n]
            B_chain[n+1] = B_chain[n]
            lls[n+1] = lls[n]
        
        log_sigmax2_mu = ((n+1)*log_sigmax2_mu + log_sigmax2_chain[n+1])/(n+2)    
        log_sigmax2_m2 = ((n+1)*log_sigmax2_m2 + log_sigmax2_chain[n+1]**2)/(n+2)
        log_sigmay2_mu = ((n+1)*log_sigmay2_mu + log_sigmay2_chain[n+1])/(n+2)    
        log_sigmay2_m2 = ((n+1)*log_sigmay2_m2 + log_sigmay2_chain[n+1]**2)/(n+2)
        A_mu = ((n+1)*A_mu + A_chain[n+1])/(n+2)    
        A_m2 = ((n+1)*A_m2 + A_chain[n+1]**2)/(n+2)
        C_mu = ((n+1)*C_mu + C_chain[n+1])/(n+2)    
        C_m2 = ((n+1)*C_m2 + C_chain[n+1]**2)/(n+2)
        B_mu = ((n+1)*B_mu + B_chain[n+1])/(n+2)    
        B_m2 = ((n+1)*B_m2 + B_chain[n+1]**2)/(n+2)
        
        if (n+1)%(n_mcmc/10) == 0 :
            print((n+1)/n_mcmc*100, "% run in", round((time()-start)/60, 1), 
                  "mins; acceptance rate =", np.round(accepted/(n+1),2))
        
        if adapt :
            if n >= int(n_mcmc*start_adapt) : 
                scales[n+1,0] = kappa*np.sqrt((log_sigmax2_m2 - log_sigmax2_mu**2))
                scales[n+1,1] = kappa*np.sqrt((log_sigmay2_m2 - log_sigmay2_mu**2))
                scales_A[n+1] = kappa*np.sqrt((A_m2 - A_mu**2))
                scales_C[n+1] = kappa*np.sqrt((C_m2 - C_mu**2))
                scales_B[n+1] = kappa*np.sqrt((B_m2 - B_mu**2))
    
    print(100*accepted/n_mcmc, "% acceptance rate")
    return log_sigmay2_chain, log_sigmax2_chain, A_chain, C_chain, B_chain, accepted


def plot_chains(log_sigmay2, log_sigmax2, A_chain, C_chain, B_chain) :
    obs_dim, lat_dim = np.shape(C_chain)[1:3]
    cov_dim = np.shape(B_chain)[-1]
    
    plt.rcParams['figure.figsize'] = (10, 2)
    plt.subplot(131)
    plt.plot(log_sigmay2)
    plt.title(r"$\log \sigma_y^2$")
    plt.grid(True)
    plt.subplot(132)
    plt.plot(log_sigmax2)
    plt.title(r"$\log \sigma_x^2$")
    plt.grid(True)
    plt.subplot(133)
    plt.plot(A_chain[:,0,0])
    plt.title(r"$A$")
    plt.grid(True)
    plt.show()
    
    plt.rcParams['figure.figsize'] = (18, 2)
    for i in range(obs_dim) :
        for j in range(cov_dim) :
            plt.subplot(1,obs_dim*cov_dim, i*cov_dim+j+1) 
            plt.plot(B_chain[:,i,j])
            plt.grid(True)
            plt.yticks(alpha=0)
            plt.title(r"$B$("+str(i+1)+str(j+1)+")")
    plt.subplots_adjust(wspace=1e-2)
    plt.show()
    
    plt.rcParams['figure.figsize'] = (10, 2)
    for i in range(obs_dim) :
        for j in range(lat_dim) :
            plt.subplot(1,obs_dim*lat_dim, i*lat_dim+j+1) 
            plt.plot(C_chain[:,i,j])
            plt.grid(True)
            plt.yticks(alpha=0)
            plt.title(r"$C$("+str(i+1)+str(j+1)+")")
    plt.subplots_adjust(wspace=1e-2)
    plt.show()



