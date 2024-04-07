
data {
    int<lower=0> E; // Number of experiments for one segment
    int<lower=0> N[E]; // Number of trials per experiment
    int<lower=0, upper=N> G[E]; // Good review count from experiments
    int<lower=1> segment[E]; // Identifier for each market segment
    real a_mean_b; // Learning from last segment
}
parameters {
    vector[E] a; // Parameter for each segment affecting the likelihood of success
    real<lower=0> sigma; // Standard deviation of effects across segments
}
model {
    // Priors
    sigma ~ exponential(1); // Prior for sigma, assuming rate 1

    // Likelihood
    for (m in 1:E) {
        vector[E] p; // Probability of success for each experiment
        p[m] = inv_logit(a[segment[m]]); // Convert logit to probability
        G[m] ~ binomial(N[m], p[m]); // Observing successes given trials and success probability
    }

    // Model for segment effects
    a ~ normal(a_mean_b, sigma); // Each segment effect is drawn from a normal distribution
    
}