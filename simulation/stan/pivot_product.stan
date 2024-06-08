data {
  real profit_obs; // Observations from sample_profit_obs function
  int<lower=1> market; // Market index
  int<lower=1> product; // Product index
  real mu_b_b_mean;
  real mu_c_b; // Kept constant during updates in this model
  // real sigma_obs;
}

parameters {
  real mu_b_b;
  real sigma_obs;
}

model {
  sigma_obs ~ exponential(1);
  mu_b_b ~ normal(mu_b_b_mean, 1);
  // Model uses fixed mu_c_b and updates mu_a and mu_b_b
  profit_obs ~ normal(pow(-1, product) * mu_b_b + pow(-1, market) * mu_c_b, sigma_obs); 
}
