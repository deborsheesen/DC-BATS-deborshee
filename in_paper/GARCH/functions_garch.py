from __future__ import division
import numpy as np, numpy.random as npr
from scipy.linalg import sqrtm
from tqdm import trange

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

def acf(y, maxlag) :
    T = len(y)
    assert maxlag < T-10
    acfs = np.zeros(maxlag+1)
    for lag in range(maxlag+1) :
        y1 = y[0:(T-lag)]
        y2 = y[lag:T]
        acfs[lag] = (np.mean(y1*y2) - np.mean(y1)*np.mean(y2))/(np.std(y1)*np.std(y2))
    return acfs

def simulate_data(T, omga, bta, alph, b, X) :
    p, q, d = len(bta), len(alph), len(b)
    
    r = max(p,q)
    sigsq = np.ones(T)
    y = np.zeros(T)
    
    for t in np.arange(0,r) :
        y[t] = np.sqrt(sigsq[t])*npr.randn() + b.dot(X[:,t])
    for t in np.arange(r,T) :
        sigsq[t] = omga + alph.dot((y[t-q:t]-b.dot(X[:,t-q:t]))**2) + bta.dot(sigsq[t-p:t]) 
        y[t] = np.sqrt(sigsq[t])*npr.randn() + b.dot(X[:,t])
        
    return sigsq, y
        
        
        
        
        
        
        
        
        
        
        