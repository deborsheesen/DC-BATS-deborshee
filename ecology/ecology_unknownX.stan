data 
{
    int<lower=0> T;
    int<lower=0> I;  # number of locations 
    int<lower=0> J;  # number of species
    int<lower=0> K;  # number of latent factors
    int<lower=0,upper=1> y[T,I,J];
    matrix[I,K] x_0;
    real<lower=0> pr_sigma;
}
parameters 
{
    real c;
    real phi;
    vector[J] alpha;
    real logsigmasq;
    vector[K] lambda[J];
    matrix[I,K] X[T];
}

model 
{
    c ~ normal(0,pr_sigma);
    phi ~ normal(0,pr_sigma);
    alpha ~ normal(0,pr_sigma);
    logsigmasq ~ normal(0,5*pr_sigma);
    for (j in 1:J) 
    {
        lambda[j] ~ normal(0,pr_sigma);
    }
    
    for (i in 1:I) 
    {
        X[1][i,:] ~ normal(c + phi*x_0[i,:], sqrt(exp(logsigmasq)));
        for (t in 2:T) 
        {
            X[t][i,:] ~ normal(c + phi*X[t-1][i,:], exp(logsigmasq/2));
        }
    }
    
    for (t in 1:T)
    {
        for (i in 1:I)
        {
            for (j in 1:J)
            {
                y[t,i,j] ~ bernoulli_logit(alpha[j] + dot_product(lambda[j],X[t][i,:]));
            }
        }
    }
    
}
