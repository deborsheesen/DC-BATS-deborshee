data 
{
    int<lower=0> T;
    int<lower=0> I;  # number of locations 
    int<lower=0> J;  # number of species
    int<lower=0> K;  # number of latent factors
    int<lower=0,upper=1> y[T,I,J];
    #real y[T,I,J];
    real<lower=0> pr_sigma;
    matrix[I,K] X[T];
}
parameters 
{
    vector[J] alpha;
    vector[K] lambda[J];
}

model 
{
    alpha ~ normal(0,pr_sigma);
    for (j in 1:J) 
    {
        lambda[j] ~ normal(0,5*pr_sigma);
    }
    
    for (t in 1:T)
    {
        for (i in 1:I)
        {
            for (j in 1:J)
            {
                y[t,i,j] ~ bernoulli_logit(alpha[j] + dot_product(lambda[j],X[t][i,:]));
                #y[t,i,j] ~ normal(alpha[j] + dot_product(lambda[j],X[t][i,:]), 1);
            }
        }
    }
    
}
