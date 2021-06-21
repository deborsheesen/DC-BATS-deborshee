data 
{
    int<lower=0> T;
    int<lower=0> I;  # number of locations 
    int<lower=0> J;  # number of species
    int<lower=0> K;  # number of latent factors
    int<lower=0,upper=1> y[T,I,J];
    matrix[I,K] x_0;
    real<lower=0> pr_sigma;
    real<lower=0> pow_obs;
    real<lower=0> pow_lat;
    real c;
    real phi;
    real logsigmasq;
}
parameters 
{
    //real c;
    //real phi;
    //real logsigmasq;
    vector[J] alpha;
    vector[K] lmbda[J];
    matrix[I,K] X[T];
}

model 
{
    // priors:
    //target += normal_lpdf(c | 0, pr_sigma);
    //target += normal_lpdf(phi | 0, pr_sigma);
    //target += normal_lpdf(logsigmasq | 0, 5*pr_sigma);
    target += normal_lpdf(alpha | 0, pr_sigma);
    for (j in 1:J) 
    {
        target += normal_lpdf(lmbda[j] | 0, pr_sigma);
    }
    
    // latent variable distribution:
    for (i in 1:I) 
    {
        target += pow_lat*normal_lpdf(X[1][i,:] | c + phi*x_0[i,:], exp(logsigmasq/2));
        for (t in 2:T) 
        {
            target += pow_lat*normal_lpdf(X[t][i,:] | c + phi*X[t-1][i,:], exp(logsigmasq/2));
        }
    }
    
    // likelihood:
    for (t in 1:T)
    {
        for (i in 1:I)
        {
            for (j in 1:J)
            {
                target += pow_obs*bernoulli_lpmf(y[t,i,j] | Phi(alpha[j] + dot_product(lmbda[j],X[t][i,:])));
            }
        }
    }
    
}
