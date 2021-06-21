data 
{
    int<lower=0> T;
    int<lower=0> m;
    int<lower=0> r;
    matrix[T,m] y;
    real<lower=0> a0;
    real<lower=0> b0;
    real bmu;
    real<lower=0> Bmu;
    real<lower=0> Bsigma;
    real<lower=0> BLambda;
    real<lower=0> power;
}
parameters 
{
    matrix[m,r] Lambda;
    matrix[T+1,m+r] h;
    matrix[T,r] f;
    row_vector[m] mu;
    row_vector<lower=0,upper=1>[m+r] phi;
    vector<lower=0>[m+r] sigma;
}

transformed parameters
{
    row_vector[m+r] phi_tr;
    phi_tr = 2*phi-1;
}

model 
{
    # priors:
    target += normal_lpdf(mu | bmu, Bmu);
    target += beta_lpdf(phi | a0, b0);
    target += gamma_lpdf(sigma | 1./2, 1./(2*Bsigma));
    for (i in 1:m) 
    {
        target += normal_lpdf(Lambda[i,:] | 0, BLambda);
    }
    for (i in 1:m)
    {
        target += normal_lpdf(h[1,i] | mu[i], sigma[i]/sqrt(1-phi_tr[i]^2));
    }
    for (i in (m+1):(m+r))
    {
        target += normal_lpdf(h[1,i] | 0, sigma[i]/sqrt(1-phi_tr[i]^2));
    }
    for (t in 2:(T+1))
    {
        target += multi_normal_lpdf(h[t,1:m] | mu + phi_tr[1:m] .* (h[t-1,1:m]-mu), diag_matrix(square(sigma[1:m])));
        target += multi_normal_lpdf(h[t,(m+1):(m+r)] | phi_tr[(m+1):(m+r)] .* h[t-1,(m+1):(m+r)], diag_matrix(square(sigma[(m+1):(m+r)])));
    }
    
    # likelihood:
    for (t in 1:T)
    {
        target += multi_normal_lpdf(f[t,:] | rep_vector(0,r), diag_matrix(to_vector(exp(h[t+1,(m+1):(m+r)]))));
        target += power*multi_normal_lpdf(y[t,:] | Lambda*to_vector(f[t,:]), diag_matrix(to_vector(exp(h[t+1,1:m]))));
    }
}

