data 
{
    int<lower=0> T;
    int<lower=0> p;
    int<lower=0> q;
    vector[T] y;
    real<lower=0> power;
}
parameters 
{
    real<lower=0> omega;
    vector<lower=0>[p] beta;
    vector<lower=0>[q] alpha;
}

model 
{
    int r = max(p,q); 
    vector[T] sigsq;
    target += gamma_lpdf(omega | 3, 10);
    target += normal_lpdf(beta | 0, 10);
    //target += inv_gamma_lpdf(sigmasq | 3, 10);
    
    sigsq[1:r] = rep_vector(1.,r);
    
    for (t in 1:r) 
    {
        target += power*normal_lpdf(y[t] | 0, sqrt(sigsq[t]));
    }
    
    for (t in (r+1):T) 
    {
        sigsq[t] = omega + alpha'*(square(y[(t-q):(t-1)]) + beta'*sigsq[(t-p):(t-1)]);
        target += power*normal_lpdf(y[t] | 0, sqrt(sigsq[t]));
    }
}




