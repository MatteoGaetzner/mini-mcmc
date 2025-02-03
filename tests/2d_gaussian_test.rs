//! Tests verifying the correctness of a Metropolis-Hastings sampler for 2D Gaussian distributions.
//!
//! This file includes two main tests:
//! 1. `test_two_d_gaussian_accept`: Checks that the sampler converges to the correct distribution.
//! 2. `test_two_d_gaussian_reject`: Confirms that the KS test rejects a wrong distribution.

// Minimal imports for the tests
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian, Normalized};
use mini_mcmc::metrohast::MetropolisHastings;
use mini_mcmc::stats;
use nalgebra as na;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;
    use mini_mcmc::ks_test::two_sample_ks_test;

    /// Randomly pick `n` rows from a `DMatrix<f64>` of shape (rows × 2).
    ///
    /// Returns a new `DMatrix<f64>` containing those rows in the same 2-column layout.
    fn pick_random_rows(matrix: &na::DMatrix<f64>, n: usize) -> na::DMatrix<f64> {
        let num_rows = matrix.nrows();

        // Ensure N is not larger than the total number of rows
        assert!(
            n <= num_rows,
            "Cannot select more rows than exist in the matrix!"
        );

        // Shuffle row indices
        let mut indices: Vec<usize> = (0..num_rows).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        // Take the first `n` indices
        let selected_indices = &indices[..n];

        // Collect the corresponding rows into a new matrix
        let selected_rows: Vec<f64> = selected_indices
            .iter()
            .flat_map(|&i| matrix.row(i).iter().copied().collect::<Vec<f64>>())
            .collect();

        na::DMatrix::from_row_slice(n, 2, &selected_rows)
    }

    /// Checks that the Metropolis-Hastings sampler produces samples consistent with
    /// the given 2D Gaussian distribution. Uses a two-sample KS test to confirm.
    #[test]
    fn test_two_d_gaussian_accept() {
        const SAMPLE_SIZE: usize = 10_000;
        const SUBSAMPLE_SIZE: usize = 1_000;
        const BURNIN: usize = 2_500;

        // Target distribution & sampler setup
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[4.0, 2.0], [2.0, 3.0]].into(),
        };
        let initial_state = vec![10.0, 12.0];
        let proposal = IsotropicGaussian::new(1.0);
        let mut mh = MetropolisHastings::new(target, proposal, initial_state);

        // Generate "true" samples from the target using Cholesky
        let chol = na::Cholesky::new(mh.target.cov).expect("Cov not positive definite");
        let z_vec: Vec<f64> = (0..(2 * SUBSAMPLE_SIZE) as i32)
            .into_par_iter()
            .map_init(rand::thread_rng, |rng, _| rng.sample(StandardNormal))
            .collect();
        let z = na::DMatrix::from_vec(2, SUBSAMPLE_SIZE, z_vec);
        let samples_target =
            na::DMatrix::from_row_slice(SUBSAMPLE_SIZE, 2, (chol.l() * z).as_slice());

        // Run MCMC, discard burn-in
        let mut samples = Vec::with_capacity(SAMPLE_SIZE + BURNIN);
        for _ in 0..(SAMPLE_SIZE + BURNIN) {
            samples.push(mh.step());
        }

        let flattened: Vec<f64> = samples[BURNIN..].iter().flatten().copied().collect();
        let samples_post_burnin = na::DMatrix::from_row_slice(SAMPLE_SIZE, 2, &flattened);

        // Randomly pick rows for KS test
        let samples_keep = pick_random_rows(&samples_post_burnin, SUBSAMPLE_SIZE);

        // Compare log probabilities via two-sample KS test
        let mut log_prob_mcmc: Vec<f64> = samples_keep
            .row_iter()
            .map(|row| mh.target.log_prob(&[row[0], row[1]]))
            .collect();
        let mut log_prob_target: Vec<f64> = samples_target
            .row_iter()
            .map(|row| mh.target.log_prob(&[row[0], row[1]]))
            .collect();

        // Quick sanity check for NaN/infinite
        assert!(
            !log_prob_mcmc
                .iter()
                .chain(log_prob_target.iter())
                .any(|x| x.is_nan() || x.is_infinite()),
            "Found infinite/NaN in log probabilities."
        );

        // KS test should accept the null (same distribution)
        let test_results = two_sample_ks_test(&mut log_prob_mcmc, &mut log_prob_target, 0.0001)
            .expect("Failed two_sample_ks_test call");

        assert!(
            !test_results.is_rejected,
            "Expected KS test to accept the null hypothesis."
        );
        assert_eq!(test_results.level, 0.0001, "KS level mismatch");

        // Validate mean & covariance
        let mean_mcmc = samples_post_burnin.row_mean();
        let cov_mcmc = stats::cov(&samples_post_burnin).unwrap();
        let good_mean = (na::DMatrix::from_column_slice(1, 2, mh.target.mean.as_slice())
            - mean_mcmc)
            .map(|x| x.abs() < 0.5);
        assert!(
            good_mean[0] && good_mean[1],
            "Mean deviation too large from target."
        );

        let good_cov = (na::DMatrix::from_column_slice(2, 2, mh.target.cov.as_slice()) - cov_mcmc)
            .map(|x| x.abs() < 0.5);
        assert!(
            good_cov[0] && good_cov[1] && good_cov[2] && good_cov[3],
            "Covariance deviation too large."
        );
    }

    /// Checks that a different (incorrect) 2D Gaussian distribution is
    /// properly rejected by the KS test.
    #[test]
    fn test_two_d_gaussian_reject() {
        const SAMPLE_SIZE: usize = 10_000;
        const SUBSAMPLE_SIZE: usize = 1_000;
        const BURNIN: usize = 2_500;

        // Correct target vs. false target
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[4.0, 2.0], [2.0, 3.0]].into(),
        };
        let false_target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[1.0, 0.0], [0.0, 1.0]].into(),
        };

        // Init MCMC with the wrong covariance
        let initial_state = vec![10.0, 12.0];
        let proposal = IsotropicGaussian::new(1.0);
        let mut mh = MetropolisHastings::new(false_target, proposal, initial_state);

        // Generate "true" samples from the correct covariance
        let chol = na::Cholesky::new(target.cov).expect("Cov not positive definite");
        let z_vec: Vec<f64> = (0..(2 * SUBSAMPLE_SIZE) as i32)
            .into_par_iter()
            .map_init(rand::thread_rng, |rng, _| rng.sample(StandardNormal))
            .collect();
        let z = na::DMatrix::from_vec(2, SUBSAMPLE_SIZE, z_vec);
        let samples_target =
            na::DMatrix::from_row_slice(SUBSAMPLE_SIZE, 2, (chol.l() * z).as_slice());

        // Run MCMC
        let mut samples = Vec::with_capacity(SAMPLE_SIZE + BURNIN);
        for _ in 0..(SAMPLE_SIZE + BURNIN) {
            samples.push(mh.step());
        }
        let flattened: Vec<f64> = samples[BURNIN..].iter().flatten().copied().collect();
        let samples_post_burnin = na::DMatrix::from_row_slice(SAMPLE_SIZE, 2, &flattened);
        let samples_keep = pick_random_rows(&samples_post_burnin, SUBSAMPLE_SIZE);

        // Compute log probabilities and run KS test
        let mut log_prob_mcmc: Vec<f64> = samples_keep
            .row_iter()
            .map(|row| mh.target.log_prob(&[row[0], row[1]]))
            .collect();
        let mut log_prob_target: Vec<f64> = samples_target
            .row_iter()
            .map(|row| mh.target.log_prob(&[row[0], row[1]]))
            .collect();

        assert!(
            !log_prob_mcmc
                .iter()
                .chain(log_prob_target.iter())
                .any(|x| x.is_nan() || x.is_infinite()),
            "Found infinite/NaN in log probabilities."
        );

        // KS test should reject the null since the distributions differ
        let test_results =
            two_sample_ks_test(&mut log_prob_mcmc, &mut log_prob_target, 0.0001).expect("");

        assert!(
            test_results.is_rejected,
            "Expected KS test to reject the null hypothesis."
        );
        assert_eq!(test_results.level, 0.0001, "KS level mismatch");
    }
}
