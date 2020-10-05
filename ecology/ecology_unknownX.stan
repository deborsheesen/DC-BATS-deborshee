data 
{
    int<lower=0> T;
    int<lower=0> I;  # number of locations 
    int<lower=0> J;  # number of species
    int<lower=0> K;  # number of latent factors
    int<lower=0,upper=1> y[T,I,J];
    matrix[I,K] x_0;
    real<lower=0> pr_sigma;
    real<lower=0> power;
    real c;
    real phi;
    real logsigmasq;
    vector[K] lmbda[J];
}
parameters 
{
    //real c;
    //real phi;
    vector[J] alpha;
    //real logsigmasq;
    //vector[K] lmbda[J];
    matrix[I,K] X[T];
}

model 
{
    target += normal_lpdf(c | 0, pr_sigma);
    target += normal_lpdf(phi | 0, pr_sigma);
    target += normal_lpdf(logsigmasq | 0, 5*pr_sigma);
    target += normal_lpdf(alpha | 0, pr_sigma);
    for (j in 1:J) 
    {
        target += normal_lpdf(lmbda[j] | 0, pr_sigma);
    }
    
    for (i in 1:I) 
    {
        target += normal_lpdf(X[1][i,:] | c + phi*x_0[i,:], exp(logsigmasq/2));
        for (t in 2:T) 
        {
            target += normal_lpdf(X[t][i,:] | c + phi*X[t-1][i,:], exp(logsigmasq/2));
        }
    }
    
    for (t in 1:T)
    {
        for (i in 1:I)
        {
            for (j in 1:J)
            {
                target += power*bernoulli_logit_lpmf(y[t,i,j] | alpha[j] + dot_product(lmbda[j],X[t][i,:]));
                // y[t,i,j] ~ bernoulli_logit(alpha[j] + dot_product(lmbda[j],X[t][i,:]));
            }
        }
    }
    
}
