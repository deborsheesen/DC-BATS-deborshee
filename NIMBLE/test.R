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
