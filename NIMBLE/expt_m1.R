## define data, constants, and initial values  
m = 1
YY = matrix(0L, nrow=t, ncol=m)
for (i in 1:t) 
{
  for (j in 1:m) 
  {
    YY[i,j] = Y[i]
  }
}
data = list(y=YY)
constants = list(t=nrow(data$y), m=m)
inits = list(a=0, sigPN=0.5, sigOE=0.5)

## build the model
stateSpaceModel=nimbleModel(stateSpaceCode, data=data, constants=constants,
                            inits=inits, check=FALSE)

## create MCMC specification for the state space model
stateSpaceMCMCconf=configureMCMC(stateSpaceModel, nodes=NULL)

## add a block pMCMC sampler for a, b, sigPN, and sigOE 
stateSpaceMCMCconf$addSampler(target=c('a', 'sigPN', 'sigOE'),
                              type='RW_PF_block', 
                              control=list(latents='x'))

## build and compile pMCMC sampler
stateSpaceMCMC = buildMCMC(stateSpaceMCMCconf)
compiledList_m1 = compileNimble(stateSpaceModel, stateSpaceMCMC, resetFunctions=TRUE)

## run compiled sampler for 10000 iterations
compiledList_m1$stateSpaceMCMC$run(1000)

## create trace plots for each parameter
par(mfrow=c(2,2))
posteriorSamps_m1 = as.mcmc(as.matrix(compiledList_m1$stateSpaceMCMC$mvSamples))
traceplot(posteriorSamps_m1[,'a'], ylab='a')
traceplot(posteriorSamps_m1[,'sigPN'], ylab='sigPN')
traceplot(posteriorSamps_m1[,'sigOE'], ylab='sigOE')

c(mean(posteriorSamps_m1[,'a']), sd(posteriorSamps_m1[,'a']))
mean(posteriorSamps_m1[,'sigPN'])
mean(posteriorSamps_m1[,'sigOE'])