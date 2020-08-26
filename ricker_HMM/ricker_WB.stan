data 
{
    int<lower=0> T;
    int<lower=0> y[T];
    real<lower=0> x_0;
    real<lower=0> power;
}
parameters 
{
    real logr;
    real<lower=0> sigma;
    real<lower=0> phi;
    real<lower=0> X[T];
}

transformed parameters
{
    real logphi;
    real logX[T];
    logphi = log(phi);
    logX = log(X);
}

model 
{
    target += normal_lpdf(logr | 0, 10);
    target += inv_gamma_lpdf(sigma | 3, 10);
    target += normal_lpdf(phi | 0, 10);
    
    target += normal_lpdf(logX[1] | log(x_0)+logr-x_0, sigma);
    
    for (t in 2:T) 
    {
        target += normal_lpdf(logX[t] | log(X[t-1])+logr-X[t-1], sigma);
    }
    
    for (t in 1:T)
    {
        target += power*poisson_lpmf(y[t] | phi*X[t]);   
    }
}

