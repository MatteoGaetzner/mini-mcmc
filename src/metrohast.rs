//! An implementation of the Metropolis-Hastings sampler, built around a generic
//! target distribution `D` and proposal distribution `Q`.

use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use nalgebra as na;
use num_traits::Float;
use rand::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::marker::PhantomData;

use crate::distributions::{ProposalDistribution, TargetDistribution};

/**
The Metropolis-Hastings sampler, parameterized by:
- `D`: the target distribution,
- `Q`: the proposal distribution.

This sampler maintains multiple parallel Markov chains (one per independent
chain) and a global random seed. It delegates the per-chain work to the
[`MHMarkovChain`] type.

# Examples

```rust
// Note: This example assumes that your crate is named `mini_mcmc`
// and that the types `Gaussian2D` and `IsotropicGaussian` are defined in
// `mini_mcmc::distributions`.
use mini_mcmc::metrohast::MetropolisHastings;
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};

let target = Gaussian2D {
    mean: [0.0, 0.0].into(),
    cov: [[1.0, 0.0], [0.0, 1.0]].into(),
};
let proposal = IsotropicGaussian::new(1.0);
let initial_state = [0.0, 0.0];
let mh = MetropolisHastings::new(target, proposal, &initial_state, 1);

// The sampler should have one chain.
assert_eq!(mh.chains.len(), 1);
```
*/
pub struct MetropolisHastings<S: Clone, T: Float, D: Clone, Q: Clone> {
    /// The target distribution we want to sample from.
    pub target: D,

    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,

    /// The set of independent Markov chains.
    pub chains: Vec<MHMarkovChain<S, T, D, Q>>,

    /// Global random seed.
    pub seed: u64,
}

#[derive(Clone)]
/**
A single Markov chain for the Metropolis-Hastings algorithm.

Each chain holds its own copy of the target and proposal distributions,
its current state, and a chain-specific random seed.
*/
pub struct MHMarkovChain<S, T, D, Q> {
    /// The target distribution we want to sample from.
    pub target: D,

    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,

    /// The current state of the Markov chain.
    pub current_state: Vec<S>,

    /// Random seed.
    pub seed: u64,

    /// Random number generator.
    pub rng: SmallRng,

    phantom: PhantomData<T>,
}

impl<S, T, D, Q> MetropolisHastings<S, T, D, Q>
where
    D: TargetDistribution<S, T> + std::clone::Clone + std::marker::Send,
    Q: ProposalDistribution<S, T> + std::clone::Clone + std::marker::Send,
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
    use mini_mcmc::metrohast::MetropolisHastings;
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

    /**
    Sets a new seed for the sampler and updates each chain's seed accordingly.

    This method allows you to control randomness and reproduce results. Each chain
    is given a unique seed by adding its index to the provided seed.

    # Arguments

    * `seed` - The new seed value.

    # Examples

    ```rust
    use mini_mcmc::metrohast::MetropolisHastings;
    use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};

    let target = Gaussian2D {
        mean: [0.0, 0.0].into(),
        cov: [[1.0, 0.0], [0.0, 1.0]].into(),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let initial_state = [0.0, 0.0];
    let mh = MetropolisHastings::new(target, proposal, &initial_state, 2).set_seed(42);
    // Each chain's seed should be 42 and 43, respectively.
    assert_eq!(mh.seed, 42);
    assert_eq!(mh.chains[0].seed, 42);
    assert_eq!(mh.chains[1].seed, 43);
    ```
    */
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        // Update each chain's seed so that they differ (if desired)
        for (i, chain) in self.chains.iter_mut().enumerate() {
            chain.seed = seed + i as u64;
            chain.rng = SmallRng::seed_from_u64(seed + i as u64)
        }
        self
    }

    /**
    Runs the sampler for a total of `n_steps` per chain, discarding the first `discard`
    samples as burn-in.

    This method runs the sampler in parallel over all chains and returns a vector of
    sample sequences (one per chain) with the burn-in samples removed.

    # Arguments

    * `n_steps` - The total number of steps to run for each chain.
    * `discard` - The number of initial samples (burn-in) to discard.

    # Returns

    A vector containing the samples (as vectors of state vectors) from each chain,
    with burn-in discarded.
    */
    pub fn run(&mut self, n_steps: usize, discard: usize) -> Vec<na::DMatrix<S>> {
        let num_threads = match std::thread::available_parallelism() {
            Ok(v) => v.get(),
            Err(_) => {
                println!("Warning: could not get number of threads; defaulting to 1.");
                1
            }
        };

        // Collect results from each chain
        let res: Vec<(Vec<S>, na::DMatrix<S>)> = self
            .chains
            .par_iter_mut()
            .with_max_len(num_threads)
            .map(|chain| {
                let samples = chain.run(n_steps);
                (chain.current_state.clone(), samples)
            })
            .collect();

        // Copy final states back into each chain
        for (i, (final_state, _)) in res.iter().enumerate() {
            self.chains[i].current_state.clone_from(final_state);
        }

        // Discard burnin and return the samples from each chain
        res.into_par_iter()
            .map(|(_, samples)| samples.rows(discard, samples.nrows() - discard).into())
            .collect()
    }

    /**
    Runs the sampler with a visual progress bar for each chain.

    This method uses the [`indicatif`] crate to display a progress bar for each chain
    while sampling. It otherwise behaves similarly to [`run`], returning the samples with
    burn-in discarded.

    # Arguments

    * `n_steps` - The total number of steps to run for each chain.
    * `discard` - The number of initial samples (burn-in) to discard.

    # Returns

    A vector containing the samples (as vectors of state vectors) from each chain,
    with burn-in discarded.
    */
    pub fn run_with_progress(&mut self, n_steps: usize, discard: usize) -> Vec<na::DMatrix<S>> {
        let num_threads = match std::thread::available_parallelism() {
            Ok(v) => v.get(),
            Err(_) => {
                println!("Warning: could not get number of threads; defaulting to 1.");
                1
            }
        };

        // Create a MultiProgress to manage multiple progress bars
        let multi = MultiProgress::new();
        let pb_style = ProgressStyle::default_bar()
            .template("{prefix} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-");

        // Collect results from each chain
        let res: Vec<(Vec<S>, na::DMatrix<S>)> = self
            .chains
            .par_iter_mut()
            .with_max_len(num_threads)
            .enumerate()
            .map(|(i, chain)| {
                // For each chain, add a new progress bar to the MultiProgress
                let pb = multi.add(indicatif::ProgressBar::new(n_steps as u64));
                pb.set_prefix(format!("Chain {i}"));
                pb.set_style(pb_style.clone());

                // Run the chain, passing in the progress bar
                let samples = chain.run_with_progress(n_steps, &pb);

                // When done, finalize this bar
                pb.finish_with_message("Done!");

                // Return final state + samples
                (chain.current_state.clone(), samples)
            })
            .collect();

        println!("Finished collecting from chains!");
        // Copy final states back into each chain
        for (i, (final_state, _)) in res.iter().enumerate() {
            self.chains[i].current_state.clone_from(final_state);
        }
        println!("Finished cloning from chains!");

        // Discard burnin and return the samples from each chain
        res.into_par_iter()
            .map(|(_, samples)| samples.rows(discard, samples.nrows() - discard).into())
            .collect()
    }
}

impl<S, T, D, Q> MHMarkovChain<S, T, D, Q>
where
    D: TargetDistribution<S, T> + Clone,
    Q: ProposalDistribution<S, T> + Clone,
    S: Clone + std::cmp::PartialEq + na::Scalar + num_traits::Zero,
    T: Float,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    const UPDATE_INTERVAL: Duration = Duration::from_millis(500);

    /**
    Creates a new Markov chain for the Metropolis-Hastings sampler.

    # Arguments

    * `target` - The target distribution from which to sample.
    * `proposal` - The proposal distribution used for generating candidate states.
    * `initial_state` - The initial state of the chain.

    # Returns

    A new instance of `MHMarkovChain`.

    # Examples

    ```rust
    use mini_mcmc::metrohast::MHMarkovChain;
    use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};

    let target = Gaussian2D {
        mean: [0.0, 0.0].into(),
        cov: [[1.0, 0.0], [0.0, 1.0]].into(),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let chain = MHMarkovChain::new(target, proposal, &[0.0, 0.0]);
    assert_eq!(chain.current_state, vec![0.0, 0.0]);
    ```
    */
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

    /**
    Runs the Markov chain for a given number of steps.

    # Arguments

    * `n_steps` - The number of iterations to run the chain.

    # Returns

    A vector containing the state of the chain after each step.
    */
    pub fn run(&mut self, n_steps: usize) -> na::DMatrix<S> {
        // let mut out = Vec::with_capacity(n_steps);
        let mut out = na::DMatrix::<S>::zeros(n_steps, self.current_state.len());
        (0..n_steps).for_each(|i| {
            out.row_mut(i).copy_from_slice(&self.step());
        });
        out
    }

    /**
    Runs the Markov chain for a given number of steps while updating a progress bar.

    This method uses the provided [`ProgressBar`] to indicate progress. The progress bar is
    updated approximately every 500 milliseconds.

    # Arguments

    * `n_steps` - The number of iterations to run the chain.
    * `pb` - A reference to a progress bar to update during the run.

    # Returns

    A vector containing the state of the chain after each step.
    */
    pub fn run_with_progress(&mut self, n_steps: usize, pb: &ProgressBar) -> na::DMatrix<S> {
        let mut out = na::DMatrix::<S>::zeros(n_steps, self.current_state.len());
        let mut accept_count = 0_usize;
        let mut last_update = Instant::now();

        pb.set_length(n_steps as u64);

        for step_idx in 0..n_steps {
            let old_state = self.current_state.clone();
            let new_state = self.step();
            if new_state != old_state {
                accept_count += 1;
            }

            out.row_mut(step_idx).copy_from_slice(&new_state);

            // Update progress bar if enough time has passed or if this is the last iteration
            if last_update.elapsed() >= Self::UPDATE_INTERVAL || step_idx + 1 == n_steps {
                let accept_rate = accept_count as f64 / (step_idx + 1) as f64;
                pb.set_position(step_idx as u64 + 1);
                pb.set_message(format!("AcceptRate={:.3}", accept_rate));
                last_update = Instant::now();
            }
        }

        out
    }

    /**
    Performs one update step of the Metropolis-Hastings algorithm.

    Proposes a new state using the proposal distribution, computes the acceptance probability
    based on the target and proposal densities, and updates the chain's current state if the
    proposal is accepted.

    # Arguments

    * `rng` - A mutable reference to a random number generator.

    # Returns

    The new state of the chain after the update step.

    # Examples

    ```rust
    use mini_mcmc::metrohast::MHMarkovChain;
    use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    let target = Gaussian2D {
        mean: [0.0, 0.0].into(),
        cov: [[1.0, 0.0], [0.0, 1.0]].into(),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let mut chain = MHMarkovChain::new(target, proposal, &[0.0, 0.0]);

    let new_state = chain.step();
    // The state should have the same dimensionality as the initial state.
    assert_eq!(new_state.len(), 2);
    ```
    */
    pub fn step(&mut self) -> Vec<S> {
        // Propose a new state based on the current state.
        let proposed: Vec<S> = self.proposal.sample(&self.current_state);

        // Compute log probabilities under the target distribution.
        let current_lp = self.target.unnorm_log_prob(&self.current_state);
        let proposed_lp = self.target.unnorm_log_prob(&proposed);

        // Compute log probabilities under the proposal distribution.
        let log_q_forward = self.proposal.log_prob(&self.current_state, &proposed);
        let log_q_backward = self.proposal.log_prob(&proposed, &self.current_state);

        // Calculate the acceptance ratio in log-space:
        //   (log p(proposed) + log q(current|proposed)) - (log p(current) + log q(proposed|current))
        let log_accept_ratio = (proposed_lp + log_q_backward) - (current_lp + log_q_forward);

        // Accept or reject based on a uniform(0,1) draw.
        let u: T = self.rng.gen();
        if log_accept_ratio > u.ln() {
            self.current_state = proposed;
        }
        self.current_state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        const SAMPLE_SIZE: usize = 100_000;
        const BURNIN: usize = 20_000;
        const N_CHAINS: usize = 4;
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
        const SAMPLE_SIZE: usize = 100_000;
        const BURNIN: usize = 100_000;
        const N_CHAINS: usize = 4;
        let seed: u64 = thread_rng().gen();

        // Setup target distribution: a bivariate Gaussian.
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[4.0, 2.0], [2.0, 3.0]].into(),
        };
        let initial_state = [0.0, 0.0];
        let proposal = IsotropicGaussian::new(1.0).set_seed(seed);
        let mut mh =
            MetropolisHastings::new(target, proposal, &initial_state, N_CHAINS).set_seed(seed);

        // --- Generate “true” samples from the target distribution ---
        // We use Cholesky decomposition to generate independent samples.
        let chol =
            na::Cholesky::new(mh.chains[0].target.cov).expect("Covariance not positive definite");
        let mut rng = SmallRng::seed_from_u64(seed);
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
    fn test_run_with_progress() {
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
        let samples_progress =
            mh_progress.run_with_progress(SAMPLE_SIZE / N_CHAINS + DISCARD, DISCARD);
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
}
