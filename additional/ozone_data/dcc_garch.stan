
data 
{
    int<lower=0> T;
    int<lower=0> dim;
    vector[dim] Y[T];
    real<lower=0> power;
    real<lower=0> sigma1_0; 
    real<lower=0> sigma2_0;
}

parameters 
{
    real<lower=0> a1;
    real<lower=0,upper=1-a1> b1;
    real<lower=0> a2;
    real<lower=0,upper=1-a2> b2;
    real<lower=0> w1;
    real<lower=0> w2;
    real<lower=0,upper=1> r;
    real mu1;
    real mu2;
}

transformed parameters 
{
    real<lower=0> sigma1[T];
    real<lower=0> sigma2[T];
    sigma1[1] = sigma1_0;
    sigma2[1] = sigma2_0;
    for (t in 2:T)   
    {
        sigma1[t] = sqrt(w1 + a1*pow(Y[t-1,1] - mu1, 2) + b1 * pow(sigma1[t-1], 2));
        sigma2[t] = sqrt(w2 + a2*pow(Y[t-1,2] - mu2, 2) + b2 * pow(sigma2[t-1], 2));
    }
}

model 
{
    // prior
    target += normal_lpdf(mu1 | 0.5, 10000);
    target += normal_lpdf(mu2 | 0.5, 10000);
    target += normal_lpdf(a1 | 0.5, 10000);
    target += normal_lpdf(b1 | 0.5, 10000);
    target += normal_lpdf(a2 | 0.5, 10000);
    target += normal_lpdf(b2 | 0.5, 10000);
    target += normal_lpdf(w1 | 1, 10000);
    target += normal_lpdf(w2 | 1, 10000);
    target += beta_lpdf(r | 1, 1);
    
    // likelihood
    for (t in 2:T)  
    {
        target += normal_lpdf(Y[t,1] | mu1, sigma1[t]);
        target += normal_lpdf(Y[t,2] | mu2 + r * sqrt(sigma2[t])/sqrt(sigma1[t])*(Y[t,1]-mu1), sigma2[t]*sqrt(1-pow(r,2)) );
    }
}











