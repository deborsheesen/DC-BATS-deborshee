library("sgmcmc")
covertype = getDataset("covertype")

X = covertype[,2:ncol(covertype)]
y = covertype[,1]
dataset = list( "X" = X, "y" = y )

d = ncol(dataset$X)
params = list( "bias" = 0, "beta" = matrix( rep( 0, d ), nrow = d ) )

logLik = function(params, dataset) {
  yEst = 1 / (1 + tf$exp( - tf$squeeze(params$bias + tf$matmul(
    dataset$X, params$beta))))
  logLik = tf$reduce_sum(dataset$y * tf$log(yEst) +
                           (1 - dataset$y) * tf$log(1 - yEst))
  return(logLik)
}

logPrior = function(params) {
  logPrior = - (tf$reduce_sum(tf$abs(params$beta)) +
                  tf$reduce_sum(tf$abs(params$bias)))
  return(logPrior)
}

stepsize = list("beta" = 2e-5, "bias" = 2e-5)

output = sgld( logLik, dataset, params, stepsize, logPrior = logPrior,
               minibatchSize = 500, nIters = 10000, seed = 13 )