data {
  real profit_obs; // Observations from sample_profit_obs function
  int<lower=1> market; // Market index
  int<lower=1> product; // Product index
  real mu_a_mean;
  real mu_c_b_mean;
  real mu_b_b; // Kept constant during updates in this model
  // real sigma_obs;
}


parameters {
  real mu_a;
  real mu_c_b;
  real sigma_obs;
}

model {
  sigma_obs ~ exponential(10);
  mu_a ~ normal(mu_a_mean, .1);
  mu_c_b ~ normal(mu_c_b_mean, .1);
  // Model uses fixed mu_b_b and updates mu_a and mu_c_b
  profit_obs ~ normal(mu_a + pow(-1, product) * mu_b_b + pow(-1, market) * mu_c_b, sigma_obs); 
}
