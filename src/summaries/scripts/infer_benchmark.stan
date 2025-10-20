data {
    int n_observations;
    vector[n_observations] x;
    real<lower=0> variance_offset;
}

parameters {
    // We focus on the tanh-transformed positive mode for better mixing and generate the actual
    // samples in `generated quantities`.
    real<lower=0, upper=1> loc;
}

transformed parameters {
    real theta_ = atanh(loc);
    real<lower=0> scale = sqrt(variance_offset - loc ^ 2);
    vector[n_observations] target_parts;
    for (i in 1:n_observations) {
        target_parts[i] = log_sum_exp(
            normal_lpdf(x[i] | loc, scale),
            normal_lpdf(x[i] | -loc, scale)
        ) - log(2);
    }
}

model {
    theta_ ~ normal(0, 1);
    target += - log1p(- loc ^ 2);  // Jacobian for the reparameterization in terms of `loc`.
    target += sum(target_parts);
}

generated quantities {
    // Randomly assign the sample to either mode.
    real theta = theta_ * (2.0 * bernoulli_rng(0.5) - 1.0);
}
