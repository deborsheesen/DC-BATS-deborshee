data 
{
    int<lower=0> T;
    int<lower=0> dim;
    int<lower=0> p;
    int<lower=0> q;
    vector[dim] Y[T];
    real<lower=0> power;
}

parameters 
{
    matrix[dim,dim] A[p];
    matrix[dim,dim] B[q];
    cholesky_factor_cov[dim] C0;
}


model 
{
    int r = max(p,q);
    matrix[dim,dim] Id = diag_matrix(rep_vector(1,dim));
    matrix[dim,dim] H[T];
    
    // prior
    for (d in 1:dim) 
    {
        for (i in 1:p)
        {
            target += normal_lpdf(A[i][d] | 0, 100);
        }
        for (j in 1:q)
        {
            target += normal_lpdf(B[j][d] | 0, 100);
        }
        target += normal_lpdf(C0[d] | 0, 100);
    }
    
    // likelihood
    for (t in 1:r)
    {
        H[t] = Id;
        // target += power*multi_normal_lpdf(Y[t] | rep_vector(0.,dim), H[t]);
        target += power*multi_student_t(Y[t] | 2, rep_vector(0.,dim), H[t]);
    }
    
    for (t in (r+1):T)
    {
        H[t] = C0*C0';
        for (i in 1:p) 
        {
            H[t] = H[t] + A[i]'*Y[t-i]*Y[t-i]'*A[i];
        }
        for (j in 1:q) 
        {
            H[t] = H[t] + B[j]'*H[t-j]*B[j];
        }
        // target += power*multi_normal_lpdf(Y[t] | rep_vector(0.,dim), H[t]);
        // target += power*multi_normal_lpdf(Y[t] | rep_vector(0.,dim), Id);
        target += power*multi_student_t(Y[t] | 2, rep_vector(0.,dim), H[t]);
    }
}













