functions
{

    vector ssm_filter_update_v(vector y, vector a, vector d, matrix Z) 
    {
        vector[num_elements(y)] v;
        v = y - Z' * a - d;
        return v;
    }
    
    matrix ssm_filter_update_F(matrix P, matrix Z, matrix H) 
    {
        matrix[rows(H), cols(H)] F;
        F = quad_form(P, Z) + H;
        return F;
    }
    
    matrix to_symmetric_matrix(matrix x) 
    {
        return 0.5 * (x + x ');
    }

    matrix ssm_filter_update_Finv(matrix P, matrix Z, matrix H) 
    {
        matrix[rows(H), cols(H)] Finv;
        Finv = inverse(ssm_filter_update_F(P, Z, H));
        return Finv;
    }

    matrix ssm_filter_update_K(matrix P, matrix Z, matrix T, matrix Finv)
    {
        matrix[rows(Z), cols(Z)] K;
        K = T * P * Z * Finv';
        return K;
    }

    real ssm_filter_update_ll(vector v, matrix Finv)
    {
        real ll;
        int p;
        p = num_elements(v);
        // det(A^{-1}) = 1 / det(A) -> log det(A^{-1}) = - log det(A)
        ll = (- 0.5 * (p * log(2 * pi()) - log_determinant(Finv) + quad_form(Finv, v) ));
        return ll;
    }

    vector ssm_filter_update_a(vector a, vector c, matrix T, vector v, matrix K) 
    {
        vector[num_elements(a)] a_new;
        a_new = T * a + K * v + c;
        return a_new;
    }

    matrix ssm_filter_update_P(matrix P, matrix Z, matrix T, matrix RQR, matrix K) 
    {
        matrix[rows(P), cols(P)] P_new;
        P_new = to_symmetric_matrix(T * P * (T - K * Z')' + RQR);
        return P_new;
    }


    real ssm_lpdf(vector[] y, vector[] d, matrix Z, matrix H,
                   vector c, matrix T, matrix R, matrix Q,
                   vector a1, matrix P1) 
    {
        real ll;
        int n;
        int m;
        int p;
        int q;
        n = size(y); // number of obs
        m = dims(Z)[1];
        p = dims(Z)[2];
        q = dims(Q)[1];
        {
            // system matrices for current iteration
            matrix[m, m] RQR;
            // result matricees for each iteration
            vector[n] ll_obs;
            vector[p] v;
            matrix[p, p] Finv;
            matrix[m, p] K;
            vector[m] a;
            matrix[m, m] P;

            a = a1;
            P = P1;
            for (t in 1:n) 
            {
                RQR = quad_form(Q, R);
                v = ssm_filter_update_v(y[t], a, d[t], Z);
                Finv = ssm_filter_update_Finv(P, Z, H);
                K = ssm_filter_update_K(P, Z, T, Finv);
                ll_obs[t] = ssm_filter_update_ll(v, Finv);
                // don't save a, P for last iteration
                if (t < n) 
                {
                    a = ssm_filter_update_a(a, c, T, v, K);
                    P = ssm_filter_update_P(P, Z, T, RQR, K);
                }
            }
            ll = sum(ll_obs);
        }
        return ll;
    }
}

data 
{
    int<lower=0> n;
    int<lower=0> m;
    int<lower=0> p;
    int<lower=0> s;
    vector[p] y[n];
    vector[s] covariate[n];
    real<lower=0> power;
    vector[m] c;
}

parameters 
{
    //real<lower=0> z1;
    //vector[p-1] z2;
    //matrix[m-1,p] Z3;
    matrix[m,p] Z;
    matrix<lower=0,upper=1>[m,m] T;
    cov_matrix[p] H;
    matrix[p,s] B;
    //vector[m] mu0;
    //cov_matrix[m] Sigma0;
}

//transformed parameters 
//{
//    matrix[m,p] Z;
//    Z[1,1] = z1;
//    for (j in 2:p)
//    {
//        Z[1,j] = z2[j-1];
//    }
//    for (i in 2:m)
//    {
//        Z[i] = Z3[i-1];
//    }
//}

transformed parameters 
{
    matrix[m,m] Ttr;
    for (i in 1:m) 
    {
        Ttr[i] = 2*T[i] - 1; 
    }
}

model 
{
    vector[p] d[n];
    matrix[m,m] R;
    matrix[m,m] Q;
    
    vector[m] mu0;
    matrix[m,m] Sigma0;
    
    R = diag_matrix(rep_vector(1,m));
    Q = diag_matrix(rep_vector(1,m));
    
    mu0 = rep_vector(0,m);
    Sigma0 = diag_matrix(rep_vector(1,m));
    
    for (t in 1:n) 
    {
        d[t] = B*covariate[t];
    }
    
    // priors
    for (i in 1:m)
    {
        target += normal_lpdf(Z[i] | 0, 100); 
        //target += uniform_lpdf(T[i] | -1, 1); 
        target += beta_lpdf(T[i] | 8, 8);
    }
    //target += normal_lpdf(mu0 | 0, 100); 
    for (i in 1:p) 
    {
        target += normal_lpdf(B[i] | 0, 100); 
    }
    target += wishart_lpdf(H | p, diag_matrix(rep_vector(1,p)));
    //target += wishart_lpdf(Sigma0 | m+1, diag_matrix(rep_vector(1,m)));
    
    // likelihood
    target += power*ssm_lpdf(y | d, Z, H, c, Ttr, R, Q, mu0, Sigma0);
}













