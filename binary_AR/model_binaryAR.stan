data 
{
    int<lower=0> T;
    int<lower=0> p;
    int<lower=0> q;
    int y[T];
    matrix[T,q] X;
    real<lower=0> power;
}
parameters 
{
    vector[p] alpha;
    real c;
    vector[q] b;
}

model 
{
    target += normal_lpdf(c | 0, 10);
    target += normal_lpdf(alpha | 0, 10);
    target += normal_lpdf(b | 0, 10);
    
    for (t in 1:p) 
    {
        target += power*bernoulli_logit_lpmf(y[t] | c + X[t]*b);
    }
    
    for (t in (p+1):T) 
    {
        target += power*bernoulli_logit_lpmf(y[t] | c + X[t]*b + alpha'*to_vector(y[(t-p+1):t]));
    }
}





