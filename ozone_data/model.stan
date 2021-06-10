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


    real ssm_lpdf(vector[] y, vector[] d, matrix[] Z, matrix[] H,
                   vector[] c, matrix[] T, matrix[] R, matrix[] Q,
                   vector a1, matrix P1) 
    {
        real ll;
        int n;
        int m;
        int p;
        int q;
        n = size(y); // number of obs
        m = dims(Z)[2];
        p = dims(Z)[3];
        q = dims(Q)[2];
        {
            // system matrices for current iteration
            vector[p] d_t;
            matrix[m, p] Z_t;
            matrix[p, p] H_t;
            vector[m] c_t;
            matrix[m, m] T_t;
            matrix[m, q] R_t;
            matrix[q, q] Q_t;
            matrix[m, m] RQR;
            // result matricees for each iteration
            vector[n] ll_obs;
            vector[m] a;
            matrix[m, m] P;
            vector[p] v;
            matrix[p, p] Finv;
            matrix[m, p] K;

            d_t = d[1];
            Z_t = Z[1];
            H_t = H[1];
            c_t = c[1];
            T_t = T[1];
            R_t = R[1];
            Q_t = Q[1];
            RQR = quad_form(Q_t, R_t);

            a = a1;
            P = P1;
            for (t in 1:n) 
            {
                if (t > 1) 
                {
                    if (size(d) > 1)
                    {
                        d_t = d[t];
                    }
                    if (size(Z) > 1)
                    {
                        Z_t = Z[t];
                    }
                    if (size(H) > 1) 
                    {
                        H_t = H[t];
                    }
                    if (size(c) > 1)
                    {
                        c_t = c[t];
                    }
                    if (size(T) > 1) 
                    {
                        T_t = T[t];
                    }
                    if (size(R) > 1) 
                    {
                        R_t = R[t];
                    }
                    if (size(Q) > 1) 
                    {
                        Q_t = Q[t];
                    }
                    if (size(R) > 1 && size(Q) > 1) 
                    {
                        RQR = quad_form(Q_t, R_t);
                    }
                }
                v = ssm_filter_update_v(y[t], a, d_t, Z_t);
                Finv = ssm_filter_update_Finv(P, Z_t, H_t);
                K = ssm_filter_update_K(P, Z_t, T_t, Finv);
                ll_obs[t] = ssm_filter_update_ll(v, Finv);
                // don't save a, P for last iteration
                if (t < n) 
                {
                    a = ssm_filter_update_a(a, c_t, T_t, v, K);
                    P = ssm_filter_update_P(P, Z_t, T_t, RQR, K);
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
    vector[p] y[n];
    real<lower=0> power;
    vector[m] c;
    vector[p] d;
    vector[m] mu0;
    cov_matrix[m] Sigma0;
}

parameters 
{
    matrix[m,p] Z;
    matrix[m,m] TT;
    cov_matrix[p] H;
}

transformed parameters
{
    matrix[m,p] Zt[n];
    cov_matrix[p] Ht[n];
    matrix[m,m] Tt[n];
    
    for (t in 1:n) 
    {
        Zt[t] = Z;
        Ht[t] = H;
        Tt[t] = TT;
    }
}

model 
{
    vector[p] dt[n];
    matrix[m,m] Qt[n];
    matrix[m,m] Rt[n];
    vector[m] ct[n];
    
    for (t in 1:n) 
    {
        Qt[t] = diag_matrix(rep_vector(1,m));
        Rt[t] = diag_matrix(rep_vector(1,m));
        ct[t] = c;
        dt[t] = d;
    }
    
    target += power*ssm_lpdf(y | dt, Zt, Ht, ct, Tt, Rt, Qt, mu0, Sigma0);
}













