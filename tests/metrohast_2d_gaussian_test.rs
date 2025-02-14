//! Tests verifying the correctness of a Metropolis-Hastings sampler for 2D Gaussian distributions.
//!
//! Instead of using a KS test, we now compare the sample means and covariance matrices.

use mini_mcmc::core::ChainRunner;
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
use mini_mcmc::metropolis_hastings::MetropolisHastings;
use nalgebra as na;

#[cfg(test)]
mod tests {
    use super::*;
    use mini_mcmc::distributions::Proposal;
    use mini_mcmc::stats::cov;

    /// Checks that the Metropolis-Hastings sampler produces samples whose
    /// mean and covariance match the given target distribution.
    #[test]
    fn test_two_d_gaussian_accept() {
        const SAMPLE_SIZE: usize = 10_000;
        const BURNIN: usize = 2_500;
        const SEED: u64 = 42;

        // Set up the target distribution.
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[4.0, 2.0], [2.0, 3.0]].into(),
        };

        // Initialize the sampler.
        let initial_state = [10.0, 12.0];
        let proposal = IsotropicGaussian::new(1.0).set_seed(SEED);
        let mut mh = MetropolisHastings::new(target, proposal, &initial_state, 1).set_seed(SEED);

        // Run the MCMC sampler (including burn-in).
        let mut samples = na::DMatrix::<f64>::zeros(SAMPLE_SIZE, 2);
        mh.run(SAMPLE_SIZE + BURNIN, BURNIN)
            .into_iter()
            .enumerate()
            .for_each(|(i, chain_samples)| {
                // Copy each batch of samples into our full sample matrix.
                samples
                    .rows_mut(i * chain_samples.nrows(), chain_samples.nrows())
                    .copy_from(&chain_samples)
            });

        // Compute sample mean and covariance (over all samples).
        let mean_mcmc = samples.row_mean();
        let cov_mcmc = cov(&samples).expect("Failed to compute covariance");

        // --- Check the sample mean ---
        // Create a 1×2 matrix from the target mean.
        let target_mean = na::DMatrix::from_column_slice(1, 2, target.mean.as_slice());
        let mean_diff = (target_mean - mean_mcmc).map(|x| x.abs());

        // We require that each component differs by less than 0.5.
        assert!(
            mean_diff[(0, 0)] < 0.5 && mean_diff[(0, 1)] < 0.5,
            "Mean deviation too large: {:?}",
            mean_diff
        );

        // --- Check the sample covariance ---
        let target_cov = na::DMatrix::from_column_slice(2, 2, target.cov.as_slice());
        let cov_diff = (target_cov - cov_mcmc).map(|x| x.abs());

        // Check each element difference is below the threshold.
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    cov_diff[(i, j)] < 0.5,
                    "Covariance deviation at ({}, {}) too large: {}",
                    i,
                    j,
                    cov_diff[(i, j)]
                );
            }
        }
    }

    /// Checks that when running the sampler with a wrong target distribution,
    /// the sample covariance (computed from the chain) is significantly different
    /// from that of the correct target.
    #[test]
    fn test_two_d_gaussian_reject() {
        const SAMPLE_SIZE: usize = 10_000;
        const BURNIN: usize = 2_500;
        const SEED: u64 = 42;

        // The correct target (only used for comparison) has a different covariance.
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[4.0, 2.0], [2.0, 3.0]].into(),
        };

        // The false target has an identity covariance.
        let false_target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[1.0, 0.0], [0.0, 1.0]].into(),
        };

        // Initialize the sampler with the false target.
        let initial_state = [10.0, 12.0];
        let proposal = IsotropicGaussian::new(1.0).set_seed(SEED);
        let mut mh =
            MetropolisHastings::new(false_target, proposal, &initial_state, 1).set_seed(SEED);

        // Run the sampler.
        let mut samples = na::DMatrix::<f64>::zeros(SAMPLE_SIZE, 2);
        mh.run(SAMPLE_SIZE + BURNIN, BURNIN)
            .into_iter()
            .enumerate()
            .for_each(|(i, chain_samples)| {
                samples
                    .rows_mut(i * chain_samples.nrows(), chain_samples.nrows())
                    .copy_from(&chain_samples)
            });

        // Compute the sample covariance from the chain.
        let cov_mcmc = cov(&samples).expect("Failed to compute covariance");

        // Compute the absolute differences between the target covariance and the samples.
        let target_cov = na::DMatrix::from_column_slice(2, 2, target.cov.as_slice());
        let cov_diff = (target_cov - cov_mcmc).map(|x| x.abs());

        // Since both targets share the same mean, we focus on covariance.
        // For the correct target, differences in each element were below 0.5.
        // Here we expect at least one element to differ by more than 1.0.
        let max_diff = cov_diff.max();
        assert!(
            max_diff > 1.0,
            "Covariance of false target samples is unexpectedly close to true target covariance. max_diff: {}",
            max_diff
        );
    }
}
