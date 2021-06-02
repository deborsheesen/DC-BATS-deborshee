library('coda')
set.seed(1)


a = 0.3
sigPN = 0.5
sigOE = 1.2
t = 100

dat = simulate_data(t, a, sigPN, sigOE)
X = dat$X
Y = dat$Y
par(mfrow=c(2,1))
plot(X, type="o")
plot(Y, type="o")



c(a, sigPN, sigOE)

sd(posteriorSamps_m1[,'sigPN'])/sd(posteriorSamps_m5[,'sigPN'])
sd(posteriorSamps_m1[,'sigOE'])/sd(posteriorSamps_m5[,'sigOE'])
sd(posteriorSamps_m1[,'a'])/sd(posteriorSamps_m5[,'a'])


c(mean(posteriorSamps_m1[,'sigPN']), mean(posteriorSamps_m5[,'sigPN']))
c(mean(posteriorSamps_m1[,'sigOE']), mean(posteriorSamps_m5[,'sigOE']))
c(mean(posteriorSamps_m1[,'a']), mean(posteriorSamps_m5[,'a']))
