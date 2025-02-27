//! # Metropolis–Hastings Sampler
//!
//! This module implements a generic Metropolis–Hastings sampler that can work with any  
//! target distribution `D` and proposal distribution `Q` that implement the corresponding  
//! traits [`Target`] and [`Proposal`]. The sampler runs multiple independent Markov chains in parallel,  
//! each initialized with the same starting state. A global seed is used to ensure reproducibility,  
//! and each chain gets a unique seed by adding its index to the global seed.
//!
//! ## Overview
//!
//! - **Target Distribution (`D`)**: Provides the (unnormalized) log-density for states via the [`Target`] trait.
//! - **Proposal Distribution (`Q`)**: Generates candidate states and computes the proposal density via the [`Proposal`] trait.
//! - **Parallel Chains**: The sampler maintains a vector of [`MHMarkovChain`] instances, each evolving independently.
//! - **Reproducibility**: The method `set_seed` assigns a unique seed to each chain based on a given global seed.
//!
//! ## Example Usage
//!
//! ```rust
//! use mini_mcmc::metropolis_hastings::MetropolisHastings;
//! use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
//!
//! // Define a 2D Gaussian target distribution with full covariance
//! let target = Gaussian2D {
//!     mean: [0.0, 0.0].into(),
//!     cov: [[1.0, 0.0], [0.0, 1.0]].into(),
//! };
//!
//! // Define an isotropic Gaussian proposal distribution (for any dimension)
//! let proposal = IsotropicGaussian::new(1.0);
//!
//! // Starting state for all chains
//! let initial_state = [0.0, 0.0];
//!
//! // Create a sampler with 1 chain
//! let mh = MetropolisHastings::new(target, proposal, &initial_state, 1);
//!
//! // Check that one chain was created
//! assert_eq!(mh.chains.len(), 1);
//! ```
//!
//! See also the documentation for [`MHMarkovChain`] and the methods below.

use nalgebra as na;
use num_traits::Float;
use rand::prelude::*;
use std::marker::{PhantomData, Send};

use crate::core::{HasChains, MarkovChain};
use crate::distributions::{Proposal, Target};

/// The Metropolis–Hastings sampler generates samples from a target distribution by  
/// using a proposal distribution to propose candidate moves and then accepting or rejecting  
/// these moves using the Metropolis–Hastings acceptance criterion.
///
/// # Type Parameters
/// - `S`: The element type for the state (typically a floating-point type).
/// - `T`: The floating-point type (e.g. `f32` or `f64`).
/// - `D`: The target distribution type. Must implement [`Target`].
/// - `Q`: The proposal distribution type. Must implement [`Proposal`].
///
/// The sampler maintains multiple independent Markov chains (each represented by [`MHMarkovChain`])
/// that are run in parallel. A global random seed is provided, and each chain’s RNG is seeded by  
/// adding the chain’s index to the global seed, ensuring reproducibility.
///
/// # Examples
///
/// ```rust
/// use mini_mcmc::metropolis_hastings::MetropolisHastings;
/// use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
///
/// let target = Gaussian2D {
///     mean: [0.0, 0.0].into(),
///     cov: [[1.0, 0.0], [0.0, 1.0]].into(),
/// };
/// let proposal = IsotropicGaussian::new(1.0);
/// let initial_state = [0.0, 0.0];
/// let mh = MetropolisHastings::new(target, proposal, &initial_state, 1);
/// assert_eq!(mh.chains.len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetropolisHastings<S: Clone, T: Float, D: Clone, Q: Clone> {
    /// The target distribution we want to sample from.
    pub target: D,
    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,
    /// The vector of independent Markov chains.
    pub chains: Vec<MHMarkovChain<S, T, D, Q>>,
    /// The global random seed.
    pub seed: u64,
}

/// A single Markov chain for the Metropolis–Hastings algorithm.
///
/// Each chain stores its own copy of the target and proposal distributions,  
/// maintains its current state, and uses a chain-specific random number generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MHMarkovChain<S, T, D, Q> {
    /// The target distribution to sample from.
    pub target: D,
    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,
    /// The current state of the chain.
    pub current_state: Vec<S>,
    /// The chain-specific random seed.
    pub seed: u64,
    /// The random number generator for this chain.
    pub rng: SmallRng,
    phantom: PhantomData<T>,
}

impl<S, T, D, Q> MetropolisHastings<S, T, D, Q>
where
    D: Target<S, T> + std::clone::Clone + Send,
    Q: Proposal<S, T> + std::clone::Clone + Send,
    T: Float + Send,
    S: Clone + std::cmp::PartialEq + Send + num_traits::Zero + std::fmt::Debug + 'static,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /**
    Constructs a new Metropolis-Hastings sampler with a given target and proposal,
    initializing each chain at `initial_state` and creating `n_chains` parallel chains.

    # Arguments

    * `target` - The target distribution from which to sample.
    * `proposal` - The proposal distribution used to generate candidate states.
    * `initial_state` - The starting state for all chains.
    * `n_chains` - The number of parallel Markov chains to run.

    # Examples

    ```rust
    use mini_mcmc::metropolis_hastings::MetropolisHastings;
    use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};

    let target = Gaussian2D {
        mean: [0.0, 0.0].into(),
        cov: [[1.0, 0.0], [0.0, 1.0]].into(),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let initial_state = [0.0, 0.0];
    let mh = MetropolisHastings::new(target, proposal, &initial_state, 1);
    assert_eq!(mh.chains.len(), 1);
    ```
    */
    pub fn new(target: D, proposal: Q, initial_state: &[S], n_chains: usize) -> Self {
        let chains = (0..n_chains)
            .map(|_| MHMarkovChain::new(target.clone(), proposal.clone(), initial_state))
            .collect();
        let seed = thread_rng().gen::<u64>();

        Self {
            target,
            proposal,
            chains,
            seed,
        }
    }

    /// Sets a new global seed and updates the seed for each chain accordingly.
    ///
    /// Each chain receives a unique seed calculated as `seed + i`, where `i` is the chain index.
    /// This method ensures reproducibility across runs and parallel chains.
    ///
    /// # Arguments
    ///
    /// * `seed` - The new global seed value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mini_mcmc::metropolis_hastings::MetropolisHastings;
    /// use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    ///
    /// let target = Gaussian2D {
    ///     mean: [0.0, 0.0].into(),
    ///     cov: [[1.0, 0.0], [0.0, 1.0]].into(),
    /// };
    /// let proposal = IsotropicGaussian::new(1.0);
    /// let initial_state = [0.0, 0.0];
    /// let mh = MetropolisHastings::new(target, proposal, &initial_state, 2)
    ///     .set_seed(42);
    /// assert_eq!(mh.chains[0].seed, 42);
    /// assert_eq!(mh.chains[1].seed, 43);
    /// ```
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        for (i, chain) in self.chains.iter_mut().enumerate() {
            let chain_seed = seed + i as u64;
            chain.seed = chain_seed;
            chain.rng = SmallRng::seed_from_u64(chain_seed)
        }
        self
    }
}

impl<S, T, D, Q> HasChains<S> for MetropolisHastings<S, T, D, Q>
where
    D: Target<S, T> + Clone + Send,
    Q: Proposal<S, T> + Clone + Send,
    T: Float + Send,
    S: Clone + PartialEq + Send + num_traits::Zero + std::fmt::Debug + 'static,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /// The concrete chain type used by the sampler.
    type Chain = MHMarkovChain<S, T, D, Q>;

    /// Returns a mutable reference to the internal vector of chains.
    ///
    /// This method allows external code to access and, if needed, modify the vector of  
    /// chains. For example, you may inspect or update individual chains using this reference.
    fn chains_mut(&mut self) -> &mut Vec<Self::Chain> {
        &mut self.chains
    }
}

impl<S, T, D, Q> MHMarkovChain<S, T, D, Q>
where
    D: Target<S, T> + Clone,
    Q: Proposal<S, T> + Clone,
    S: Clone + std::cmp::PartialEq + na::Scalar + num_traits::Zero,
    T: Float,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /// Creates a new Metropolis–Hastings chain.
    ///
    /// # Arguments
    /// * `target` - The target distribution.
    /// * `proposal` - The proposal distribution.
    /// * `initial_state` - The starting state for the chain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mini_mcmc::metropolis_hastings::MHMarkovChain;
    /// use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    ///
    /// let target = Gaussian2D {
    ///     mean: [0.0, 0.0].into(),
    ///     cov: [[1.0, 0.0], [0.0, 1.0]].into(),
    /// };
    /// let proposal = IsotropicGaussian::new(1.0);
    /// let chain = MHMarkovChain::new(target, proposal, &[0.0, 0.0]);
    /// assert_eq!(chain.current_state, vec![0.0, 0.0]);
    /// ```
    pub fn new(target: D, proposal: Q, initial_state: &[S]) -> Self {
        let seed = thread_rng().gen::<u64>();
        Self {
            target,
            proposal,
            current_state: initial_state.to_vec(),
            seed,
            rng: SmallRng::seed_from_u64(seed),
            phantom: PhantomData,
        }
    }
}

impl<S, T, D, Q> MarkovChain<S> for MHMarkovChain<S, T, D, Q>
where
    D: Target<S, T> + Clone,
    Q: Proposal<S, T> + Clone,
    S: Clone + PartialEq + na::Scalar + num_traits::Zero,
    T: Float,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /// Performs one Metropolis–Hastings update step.
    ///
    /// A new candidate state is proposed using the proposal distribution.  
    /// The unnormalized log-density of the current and proposed states is computed,  
    /// along with the corresponding proposal densities. The acceptance ratio in log-space  
    /// is calculated as:
    ///
    /// \[
    /// \log \alpha = \left[\log p(\text{proposed}) + \log q(\text{current} \mid \text{proposed})\right]
    ///               - \left[\log p(\text{current}) + \log q(\text{proposed} \mid \text{current})\right]
    /// \]
    ///
    /// A uniform random number is drawn, and if \(\log(\text{Uniform}(0,1))\) is less than  
    /// \(\log \alpha\), the proposed state is accepted. Otherwise, the current state is retained.
    ///
    /// The method returns a reference to the updated state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mini_mcmc::metropolis_hastings::MHMarkovChain;
    /// use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    /// use mini_mcmc::core::MarkovChain;
    ///
    /// let target = Gaussian2D {
    ///     mean: [0.0, 0.0].into(),
    ///     cov: [[1.0, 0.0], [0.0, 1.0]].into(),
    /// };
    /// let proposal = IsotropicGaussian::new(1.0);
    /// let mut chain = MHMarkovChain::new(target, proposal, &[0.0, 0.0]);
    /// let new_state = chain.step();
    /// assert_eq!(new_state.len(), 2);
    /// ```
    fn step(&mut self) -> &Vec<S> {
        let proposed: Vec<S> = self.proposal.sample(&self.current_state);
        let current_lp = self.target.unnorm_log_prob(&self.current_state);
        let proposed_lp = self.target.unnorm_log_prob(&proposed);
        let log_q_forward = self.proposal.log_prob(&self.current_state, &proposed);
        let log_q_backward = self.proposal.log_prob(&proposed, &self.current_state);
        let log_accept_ratio = (proposed_lp + log_q_backward) - (current_lp + log_q_forward);
        let u: T = self.rng.gen();
        if log_accept_ratio > u.ln() {
            self.current_state = proposed;
        }
        &self.current_state
    }

    /// Returns a reference to the current state of the chain.
    fn current_state(&self) -> &Vec<S> {
        &self.current_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ChainRunner;
    use crate::distributions::Normalized;
    use crate::ks_test::TotalF64;
    use rand_distr::StandardNormal;

    use crate::{
        distributions::{Gaussian2D, IsotropicGaussian},
        stats,
    };
    use nalgebra as na;

    #[test]
    fn four_chains_test() {
        const SAMPLE_SIZE: usize = 20_000;
        const BURNIN: usize = 5_000;
        const N_CHAINS: usize = 8;
        const SEED: u64 = 42;

        // Target distribution & sampler setup
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[4.0, 2.0], [2.0, 3.0]].into(),
        };
        let initial_state = [0.0, 0.0];
        let proposal = IsotropicGaussian::<f64>::new(1.0).set_seed(SEED);
        let mut mh =
            MetropolisHastings::new(target, proposal, &initial_state, N_CHAINS).set_seed(SEED);

        // Generate "true" samples from the target using Cholesky
        let chol = na::Cholesky::new(mh.chains[0].target.cov).expect("Cov not positive definite");
        let mut rng = SmallRng::seed_from_u64(SEED);
        let z_vec: Vec<f64> = (0..(2 * SAMPLE_SIZE))
            .map(|_| rng.sample(StandardNormal))
            .collect();

        let z = na::DMatrix::from_vec(2, SAMPLE_SIZE, z_vec);
        let samples_target = na::DMatrix::from_row_slice(SAMPLE_SIZE, 2, (chol.l() * z).as_slice());

        // Run MCMC, discard burn-in
        let mut samples = na::DMatrix::<f64>::zeros(SAMPLE_SIZE, 2);
        mh.run(SAMPLE_SIZE / N_CHAINS + BURNIN, BURNIN)
            .into_iter()
            .enumerate()
            .for_each(|(i, chain_samples)| {
                dbg!(i, chain_samples.row_mean());
                samples
                    .rows_mut(i * chain_samples.nrows(), chain_samples.nrows())
                    .copy_from(&chain_samples)
            });

        // Validate mean & covariance
        let mean_mcmc = samples.row_mean();
        let cov_mcmc = stats::cov(&samples).unwrap();

        // Compare log probabilities via two-sample KS test
        let log_prob_mcmc: Vec<f64> = samples
            .row_iter()
            .map(|row| mh.target.log_prob(&[row[0], row[1]]))
            .collect();
        let log_prob_target: Vec<f64> = samples_target
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

        let good_mean = (na::DMatrix::from_column_slice(1, 2, mh.target.mean.as_slice())
            - mean_mcmc)
            .map(|x| x.abs() < 0.5);
        assert!(
            good_mean[0] && good_mean[1],
            "Mean deviation too large from target."
        );

        dbg!(mh.target.cov, &cov_mcmc);
        let good_cov = (na::DMatrix::from_column_slice(2, 2, mh.target.cov.as_slice()) - cov_mcmc)
            .map(|x| x.abs() < 0.5);
        assert!(
            good_cov[0] && good_cov[1] && good_cov[2] && good_cov[3],
            "Covariance deviation too large."
        );
    }

    //
    // This test uses the MH sampler to generate samples, computes the log‐probabilities
    // of each sample (under both the target and an “exact” sampler), and then runs the
    // KS test on these log probabilities. It prints both the internal and external KS test results.
    //
    #[test]
    fn test_mh_ks_comparison() {
        // --- Setup MH sampler parameters ---
        const SAMPLE_SIZE: usize = 10_000;
        const BURNIN: usize = 2_000;
        const N_CHAINS: usize = 8;
        const SEED: u64 = 42;

        // Setup target distribution: a bivariate Gaussian.
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[4.0, 2.0], [2.0, 3.0]].into(),
        };
        let initial_state = [0.0, 0.0];
        let proposal = IsotropicGaussian::new(1.0).set_seed(SEED);
        let mut mh =
            MetropolisHastings::new(target, proposal, &initial_state, N_CHAINS).set_seed(SEED);

        // --- Generate “true” samples from the target distribution ---
        // We use Cholesky decomposition to generate independent samples.
        let chol =
            na::Cholesky::new(mh.chains[0].target.cov).expect("Covariance not positive definite");
        let mut rng = SmallRng::seed_from_u64(SEED);
        let z_vec: Vec<f64> = (0..(2 * SAMPLE_SIZE))
            .map(|_| rng.sample(StandardNormal))
            .collect();
        let z = na::DMatrix::from_vec(2, SAMPLE_SIZE, z_vec);
        let samples_target = na::DMatrix::from_row_slice(SAMPLE_SIZE, 2, (chol.l() * z).as_slice());

        // --- Run the MH sampler ---
        // We run enough steps so that after discarding burn-in we have SAMPLE_SIZE samples.
        let mut samples_mcmc = na::DMatrix::<f64>::zeros(SAMPLE_SIZE, 2);
        mh.run(SAMPLE_SIZE / N_CHAINS + BURNIN, BURNIN)
            .into_iter()
            .enumerate()
            .for_each(|(i, chain_samples)| {
                samples_mcmc
                    .rows_mut(i * chain_samples.nrows(), chain_samples.nrows())
                    .copy_from(&chain_samples)
            });

        // --- Compute log probabilities for both sets of samples ---
        // (We assume your target distribution provides log_prob)
        let log_prob_mcmc: Vec<f64> = samples_mcmc
            .row_iter()
            .map(|row| mh.target.log_prob(&[row[0], row[1]]))
            .collect();
        let log_prob_target: Vec<f64> = samples_target
            .row_iter()
            .map(|row| mh.target.log_prob(&[row[0], row[1]]))
            .collect();

        // --- Convert the log probabilities to TotalF64 so they can be sorted ---
        let log_prob_mcmc_total: Vec<TotalF64> =
            log_prob_mcmc.iter().copied().map(TotalF64).collect();
        let log_prob_target_total: Vec<TotalF64> =
            log_prob_target.iter().copied().map(TotalF64).collect();

        // --- Run the internal KS test ---
        // Here the chosen significance level (0.05) is arbitrary.
        let res_internal: crate::ks_test::TestResult =
            crate::ks_test::two_sample_ks_test(&log_prob_mcmc_total, &log_prob_target_total, 0.05)
                .expect("Internal KS test failed");

        // --- Run the external KS test ---
        // The external function may use a different significance level (here 0.95).
        let res_external =
            kolmogorov_smirnov::test(&log_prob_mcmc_total, &log_prob_target_total, 0.95);

        // --- Print both results for manual verification ---
        println!(
            "Internal KS test result:\n  statistic: {}\n  p_value: {}\n  is_rejected: {}",
            res_internal.statistic, res_internal.p_value, res_internal.is_rejected
        );
        println!(
            "External KS test result:\n  statistic: {}\n  p_value: {}\n  is_rejected: {}",
            res_external.statistic,
            1.0 - res_external.reject_probability,
            res_external.is_rejected
        );

        assert!(
            (res_internal.p_value - 1.0 + res_external.reject_probability).abs() < 0.001,
            "Expected p-values to be close to each other."
        );
    }

    #[test]
    fn test_run_progress() {
        // We'll do a small run to keep the test fast
        const SAMPLE_SIZE: usize = 2000;
        const DISCARD: usize = 500;
        const N_CHAINS: usize = 2;
        const SEED: u64 = 42;

        // Setup a small Gaussian2D target and an isotropic proposal
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[1.0, 0.0], [0.0, 1.0]].into(),
        };
        let proposal = IsotropicGaussian::<f32>::new(0.5).set_seed(SEED);
        let initial_state = [1.0, 1.0];

        // Create the MH sampler with multiple chains
        let mut mh_progress =
            MetropolisHastings::new(target, proposal.clone(), &initial_state.clone(), N_CHAINS)
                .set_seed(SEED);

        // Run with progress bars
        println!("STARTING RUNNING WITH PROGRESS");
        let samples_progress = mh_progress.run_progress(SAMPLE_SIZE / N_CHAINS + DISCARD, DISCARD);
        println!("FINISHED RUNNING WITH PROGRESS");

        // Basic checks
        assert_eq!(samples_progress.len(), N_CHAINS);

        for chain_samples in &samples_progress {
            assert_eq!(chain_samples.nrows(), SAMPLE_SIZE / 2);
        }

        // Compare to normal run
        let mut mh_normal =
            MetropolisHastings::new(target, proposal, &initial_state, N_CHAINS).set_seed(SEED);
        let samples_normal = mh_normal.run(SAMPLE_SIZE / N_CHAINS + DISCARD, DISCARD);

        // Check that shape is the same
        assert_eq!(samples_normal.len(), N_CHAINS);
        for chain_samples in &samples_normal {
            assert_eq!(chain_samples.nrows(), SAMPLE_SIZE / 2);
        }

        // Compare means across the two runs.
        // We'll just do a naive check that they're "similar".
        for (sp_chain, sn_chain) in samples_progress.iter().zip(samples_normal.iter()) {
            let sp_mean = sp_chain.row_mean();
            let sn_mean = sn_chain.row_mean();
            assert!(
                (sp_mean[0] - sn_mean[0]).abs() < 0.5,
                "Means differ more than expected on dimension 0"
            );
            assert!(
                (sp_mean[1] - sn_mean[1]).abs() < 0.5,
                "Means differ more than expected on dimension 1"
            );
        }
    }

    #[test]
    fn readme_test() {
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[1.0, 0.0], [0.0, 1.0]].into(),
        };
        let proposal = IsotropicGaussian::new(1.0);
        let initial_state = [0.0, 0.0];

        // Create a MH sampler with 4 parallel chains
        let mut mh = MetropolisHastings::new(target, proposal, &initial_state, 4);

        // Run the sampler for 1,000 steps, discarding the first 100 as burn-in
        let samples = mh.run(1000, 100);

        // We should have 900 * 4 = 3600 samples
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].nrows(), 900);
    }
}
