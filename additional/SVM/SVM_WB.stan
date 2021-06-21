data 
{
    int<lower=0> T;  
    vector[T] y;    
    real<lower=0> power_ll;
    real<lower=0> power_lat;
    real<lower=0> power_pr;
}
parameters
{
    real mu;                    
    real<lower=-1,upper=1> phi;  
    real<lower=0> sigma;         
    vector[T] h;                 
}
model 
{
    target += power_pr*cauchy_lpdf(mu | 0, 10);
    target += power_pr*uniform_lpdf(phi | -1, 1);
    target += power_pr*cauchy_lpdf(sigma | 0, 10);
    target += power_lat*normal_lpdf(h[1] | mu, sigma/sqrt(1 - phi*phi));
    for (t in 2:T)
    {
        target += power_lat*normal_lpdf(h[t] | mu + phi*(h[t-1]-mu), sigma);
    }
    for (t in 1:T)
    {
        target += power_ll*normal_lpdf(y[t] | 0, exp(h[t]/2));
    }
}