data {
  real profit_obs; // Observations from sample_profit_obs function
  int<lower=1> market; // Market index
  int<lower=1> product; // Product index
  real mu_p_b_mean;
  real mu_m_b; // Kept constant during updates in this model
  // real sigma_obs;
}

parameters {
  real mu_p_b;
  real sigma_obs;
}

model {
  sigma_obs ~ exponential(1);
  mu_p_b ~ normal(mu_p_b_mean, 1);
  // Model uses fixed mu_m_b and updates mu_a and mu_p_b
  profit_obs ~ normal(pow(-1, product) * mu_p_b/2 + pow(-1, market) * mu_m_b/2, sigma_obs); 
}
