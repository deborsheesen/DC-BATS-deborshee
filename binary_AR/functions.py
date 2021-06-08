from __future__ import division
import numpy as np, numpy.random as npr
from scipy.linalg import sqrtm
from tqdm import trange

def generate_X(T, k) :
    p1, q1 = 10, 10
    r = max(p1,q1)
    X = np.zeros((T,k))
    
    for i in range(k) :

        z = np.zeros(T)
        omega = npr.rand()
        beta = npr.rand(p1)/(1.2*p1)
        alpha = npr.rand(q1)/(1.2*q1)

        sigsq, z = np.ones(T), np.zeros(T)
        for t in np.arange(0,r) :
            z[t] = np.sqrt(sigsq[t])*npr.randn() 
        for t in np.arange(r,T) :
            sigsq[t] = omega + beta.dot(sigsq[t-p1:t]) + alpha.dot((z[t-q1:t]**2))
            z[t] = np.sqrt(sigsq[t])*npr.randn()
        
        X[:,i] = sigsq
    
    return X

def double_parallel_MC(samples, mle) :
    m, n_mcmc, p = np.shape(samples)
    assert m, p == np.shape(mle)
    
    mus = np.zeros((m,p))
    Sigmas = np.zeros((m,p,p))
    Sigmas_sqrt_inv = np.zeros((m,p,p))

    for i in range(m) :
        mus[i] = np.mean(samples[i],0)
        Sigmas[i] = np.cov(samples[i].transpose())
        Sigmas_sqrt_inv[i] = np.linalg.inv(sqrtm(Sigmas[i]))
        
    mu_bar = np.mean(mus,0)
    Sigma_bar = np.mean(Sigmas,0)
    sqrt_Sigma_bar = sqrtm(Sigma_bar)
    
    samples_dpMC = np.zeros((m,n_mcmc,p))
    for j in trange(n_mcmc) :
        for i in range(m) :
            samples_dpMC[i,j] = mu_bar + sqrt_Sigma_bar.dot(Sigmas_sqrt_inv[i]).dot(samples[i,j]-mle[i])
            
    return samples_dpMC