data {
    int<lower=0> E; 
    int<lower=0, upper=1> profit_r[E]; 
    int<lower=1> market[E]; 
    int<lower=1> product[E];
    real rev_mean_b;  
    vector[E] cost_b;
    real<lower=0> UB; 
}
parameters {
    vector[E] rev_b;
    real<lower=0> sigma_rev;
}
model {
    sigma_rev ~ exponential(UB); 
    rev_b ~ normal(rev_mean_b, sigma_rev);

    for (e in 1:E) {
        vector[E] p; 
        p[e] = inv_logit(rev_b[market[e]] - cost_b[product[e]]); 
        profit_r[e] ~ binomial(1, p[e]); 
    }
}
generated quantities {
    vector[E] updated_rev_mean_b;
    for (e in 1:E) {
        updated_rev_mean_b[e] = normal_rng(rev_b[e], sigma_rev);
    }
}
