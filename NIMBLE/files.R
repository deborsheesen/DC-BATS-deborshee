## load the nimble library and set seed
library('nimble')
library('nimbleSMC')

simulate_data = function(t, a, sigPN, sigOE)
{
  X = rep(0,T)
  Y = rep(0,T)
  X[1] = rnorm(1)
  Y[1] = sigOE*exp(X[1]/2)*rnorm(1)
  for (i in 2:t) 
  {
    X[i] = a*X[i-1] + sigPN*rnorm(1)
    Y[i] = sigOE*exp(X[i]/2)*rnorm(1)
  }
  return(list(X=X,Y=Y))
}

## define the model
stateSpaceCode=nimbleCode(
  {
    a ~ dunif(-0.99, 0.99)
    sigPN ~ dinvgamma(shape=3, scale=3)
    sigOE ~ dinvgamma(shape=3, scale=3)
    for (j in 1:m)
    {
      x[1,j] ~ dnorm(0, sd=1)
      y[1,j] ~ dnorm(0, sd=sigOE*exp(x[1,j]/2))
      for (i in 2:t) 
      {
        x[i,j] ~ dnorm(a*x[i-1,j], sd=sigPN)
        y[i,j] ~ dnorm(0, sd=sigOE*exp(x[i,j]/2))
      }
    }
  })



## build bootstrap filter and compile model and filter
#bootstrapFilter=buildBootstrapFilter(stateSpaceModel, nodes='x')
#compiledList=compileNimble(stateSpaceModel, bootstrapFilter)

## run compiled filter with 10,000 particles.  
## note that the bootstrap filter returns an estimate of the log-likelihood of the model.
#compiledList$bootstrapFilter$run(10000)



