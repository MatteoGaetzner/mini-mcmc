//! Tests verifying the correctness of a Metropolis-Hastings sampler for 2D Gaussian distributions.
//!
//! Instead of using a KS test, we now compare the sample means and covariance matrices.

#[cfg(test)]
mod tests {
    use mini_mcmc::core::ChainRunner;
    use mini_mcmc::distributions::Proposal;
    use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use mini_mcmc::metropolis_hastings::MetropolisHastings;
    use ndarray::{arr1, arr2, Axis};
    use ndarray_stats::CorrelationExt;
    use ndarray_stats::QuantileExt;

    // Shared constants.
    const SAMPLE_SIZE: usize = 10_000;
    const BURNIN: usize = 2_500;
    const SEED: u64 = 42;
    const INITIAL_STATE: [f64; 2] = [10.0, 12.0];

    /// Runs the Metropolis-Hastings sampler with the provided target distribution,
    /// and returns the samples reshaped into a (SAMPLE_SIZE, 2) array.
    fn run_sampler(target: &Gaussian2D<f64>) -> ndarray::Array2<f64> {
        let proposal = IsotropicGaussian::new(1.0).set_seed(SEED);
        let mut mh = MetropolisHastings::new(target.clone(), proposal, vec![INITIAL_STATE.into()])
            .seed(SEED);
        let samples = mh.run(SAMPLE_SIZE, BURNIN).unwrap();
        samples.to_shape((SAMPLE_SIZE, 2)).unwrap().to_owned()
    }

    /// Checks that the sampler produces samples with mean and covariance close to the target.
    #[test]
    fn test_two_d_gaussian_accept() {
        // Set up the target distribution.
        let target = Gaussian2D {
            mean: arr1(&[0.0, 0.0]),
            cov: arr2(&[[4.0, 2.0], [2.0, 3.0]]),
        };

        let samples = run_sampler(&target);

        // Compute sample mean and covariance.
        let mean_mcmc = samples.mean_axis(Axis(0)).unwrap();
        let cov_mcmc = samples.t().cov(1.0).unwrap();

        // Check the sample mean (each component should differ by less than 0.5).
        let mean_diff = (mean_mcmc - target.mean).abs();
        assert!(
            mean_diff[0] < 0.5 && mean_diff[1] < 0.5,
            "Mean deviation too large: {}",
            mean_diff
        );

        // Compute the maximum absolute difference in covariance.
        let max_diff = *(cov_mcmc - target.cov).abs().max().unwrap();

        assert!(
            max_diff < 0.5,
            "Covariance of false target samples is unexpectedly close to true target covariance. max_diff: {}",
            max_diff
        );
    }

    /// Checks that when using a false target distribution, the sample covariance differs
    /// significantly from that of the correct target.
    #[test]
    fn test_two_d_gaussian_reject() {
        // The correct target (for comparison) has the following covariance.
        let target = Gaussian2D {
            mean: arr1(&[0.0, 0.0]),
            cov: arr2(&[[4.0, 2.0], [2.0, 3.0]]),
        };

        // The false target uses an identity covariance.
        let false_target = Gaussian2D {
            mean: arr1(&[0.0, 0.0]),
            cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        };

        let samples = run_sampler(&false_target);
        let cov_mcmc = samples.t().cov(1.0).unwrap();

        // Compute the maximum absolute difference in covariance.
        let max_diff = *(cov_mcmc - target.cov).abs().max().unwrap();

        // Expect at least one element to differ by more than 1.0.
        assert!(
            max_diff > 1.0,
            "Covariance of false target samples is unexpectedly close to true target covariance. max_diff: {}",
            max_diff
        );
    }
}
