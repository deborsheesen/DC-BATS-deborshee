data 
{
    int<lower=0> T;
    int<lower=0> p;
    int<lower=0> q;
    int<lower=0> r;
    int<lower=0> nu;
    row_vector[T] y;
    real<lower=0> power;
}
parameters 
{
    real<lower=0> omega;
    vector<lower=0>[p] beta;
    vector<lower=0>[q] alpha;
    real mu;
}

model 
{
    vector[T] sigsq;
    target += gamma_lpdf(omega | 3, 10);
    target += normal_lpdf(beta | 0, 10);
    target += normal_lpdf(alpha | 0, 10);
    target += normal_lpdf(mu | 0, 10);
    //target += inv_gamma_lpdf(sigmasq | 3, 10);
    
    sigsq[1:r] = rep_vector(1.,r);
    
    for (t in 1:r) 
    {
        target += power*normal_lpdf(y[t] | mu, sqrt(sigsq[t]));
    }
    
    for (t in (r+1):T) 
    {
        sigsq[t] = omega + alpha'*(square(y[(t-q):(t-1)]- rep_row_vector(mu,q))') + beta'*sigsq[(t-p):(t-1)];
        target += power*student_t_lpdf(y[t] | nu, mu, sqrt(sigsq[t]));
    }
}





