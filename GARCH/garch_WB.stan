data 
{
    int<lower=0> T;
    int<lower=0> p;
    int<lower=0> q;
    int<lower=0> r;
    int<lower=0> d;
    row_vector[T] y;
    matrix[d,T] X;
    real<lower=0> power;
}
parameters 
{
    real<lower=0> omega;
    vector<lower=0>[p] beta;
    vector<lower=0>[q] alpha;
    vector[d] b;
}

model 
{
    vector[T] sigsq;
    target += gamma_lpdf(omega | 3, 10);
    target += normal_lpdf(beta | 0, 10);
    target += normal_lpdf(alpha | 0, 10);
    target += normal_lpdf(b | 0, 10);
    //target += inv_gamma_lpdf(sigmasq | 3, 10);
    
    sigsq[1:r] = rep_vector(1.,r);
    
    for (t in 1:r) 
    {
        target += power*normal_lpdf(y[t] | b'*X[:,t], sqrt(sigsq[t]));
    }
    
    for (t in (r+1):T) 
    {
        sigsq[t] = omega + alpha'*(square(y[(t-q):(t-1)]-b'*X[:,(t-q):(t-1)])') + beta'*sigsq[(t-p):(t-1)];
        target += power*normal_lpdf(y[t] | b'*X[:,t], sqrt(sigsq[t]));
    }
}





