//! An implementation of the Metropolis-Hastings sampler, built around a generic
//! target distribution `D` and proposal distribution `Q`.

use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::distributions::{ProposalDistribution, TargetDistribution};

// ChainResult: (last state of Markov chain, series of states)
type ChainResult = (Vec<f64>, Vec<Vec<f64>>);

/// The Metropolis-Hastings sampler, parameterized by:
/// - `D`: the target distribution,
/// - `Q`: the proposal distribution.
///
/// Maintains the current state of the chain, as well as an RNG.
pub struct MetropolisHastings<D: Clone, Q: Clone> {
    /// The target distribution we want to sample from.
    pub target: D,

    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,

    /// Parallel Markov chains.
    pub chains: Vec<MHMarkovChain<D, Q>>,

    /// Random seed.
    pub seed: u64,
}

#[derive(Clone)]
pub struct MHMarkovChain<D, Q> {
    /// The target distribution we want to sample from.
    pub target: D,

    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,

    /// The current state of the Markov chain.
    pub current_state: Vec<f64>,

    /// Random seed.
    pub seed: u64,
}

impl<D, Q> MetropolisHastings<D, Q>
where
    D: TargetDistribution + std::clone::Clone + std::marker::Send,
    Q: ProposalDistribution + std::clone::Clone + std::marker::Send,
{
    /// Constructs a new Metropolis-Hastings sampler with a given target and proposal,
    /// initializing the chain at `initial_state`.
    pub fn new(target: D, proposal: Q, initial_state: Vec<f64>, n_chains: usize) -> Self {
        let chains = (0..n_chains)
            .map(|_| MHMarkovChain::new(target.clone(), proposal.clone(), initial_state.clone()))
            .collect();

        Self {
            target,
            proposal,
            chains,
            seed: thread_rng().gen(),
        }
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        // Update each chain's seed so that they differ (if desired)
        for (i, chain) in self.chains.iter_mut().enumerate() {
            chain.seed = seed + i as u64;
        }
        self
    }

    pub fn run(&mut self, n_steps: usize, discard: usize) -> Vec<Vec<Vec<f64>>> {
        let num_threads = match std::thread::available_parallelism() {
            Ok(v) => v.get(),
            Err(_) => {
                println!("Warning: could not get number of threads; defaulting to 1.");
                1
            }
        };

        // Collect results from each chain
        let res: Vec<ChainResult> = self
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
            .map(|(_, samples)| samples[discard..].to_vec())
            .collect()
    }

    pub fn run_with_progress(&mut self, n_steps: usize, discard: usize) -> Vec<Vec<Vec<f64>>> {
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
        let res: Vec<ChainResult> = self
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

        // Copy final states back into each chain
        for (i, (final_state, _)) in res.iter().enumerate() {
            self.chains[i].current_state.clone_from(final_state);
        }

        // Discard burnin and return the samples from each chain
        res.into_par_iter()
            .map(|(_, samples)| samples[discard..].to_vec())
            .collect()
    }
}

impl<D, Q> MHMarkovChain<D, Q>
where
    D: TargetDistribution + Clone,
    Q: ProposalDistribution + Clone,
{
    pub fn new(target: D, proposal: Q, initial_state: Vec<f64>) -> Self {
        Self {
            target,
            proposal,
            current_state: initial_state,
            seed: thread_rng().gen(),
        }
    }

    pub fn run(&mut self, n_steps: usize) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(n_steps);
        let mut rng = SmallRng::seed_from_u64(self.seed);
        (0..n_steps).for_each(|_| out.push(self.step(&mut rng)));
        out
    }

    pub fn run_with_progress(&mut self, n_steps: usize, pb: &ProgressBar) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(n_steps);
        let mut rng = SmallRng::seed_from_u64(self.seed);

        let mut accept_count = 0_usize;

        pb.set_length(n_steps as u64);

        const UPDATE_INTERVAL: Duration = Duration::from_millis(500);
        let mut last_update = Instant::now();

        for step_idx in 0..n_steps {
            let old_state = self.current_state.clone();
            let new_state = self.step(&mut rng);

            if new_state != old_state {
                accept_count += 1;
            }

            out.push(new_state);

            // Check if 0.5 s have passed or if we’re at the final iteration
            if last_update.elapsed() >= UPDATE_INTERVAL || step_idx + 1 == n_steps {
                // Compute acceptance rate
                let accept_rate = accept_count as f64 / (step_idx + 1) as f64;

                // Move progress bar to current step
                pb.set_position(step_idx as u64 + 1);
                pb.set_message(format!("AcceptRate={:.3}", accept_rate));

                // Reset the timer
                last_update = Instant::now();
            }
        }

        out
    }

    /// Performs one Metropolis-Hastings update. Proposes a new state, computes the
    /// acceptance probability, and returns the (potentially updated) current state.
    ///
    /// # Returns
    ///
    /// The current state of the chain after this iteration.
    pub fn step(&mut self, rng: &mut SmallRng) -> Vec<f64> {
        // Propose a new state based on the current state.
        let proposed = self.proposal.sample(&self.current_state);

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
        let u: f64 = rng.gen();
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
        let initial_state = vec![0.0, 0.0];
        let proposal = IsotropicGaussian::new(1.0).set_seed(SEED);
        let mut mh =
            MetropolisHastings::new(target, proposal, initial_state, N_CHAINS).set_seed(SEED);

        // Generate "true" samples from the target using Cholesky
        let chol = na::Cholesky::new(mh.chains[0].target.cov).expect("Cov not positive definite");
        let mut rng = SmallRng::seed_from_u64(SEED);
        let z_vec: Vec<f64> = (0..(2 * SAMPLE_SIZE))
            .map(|_| rng.sample(StandardNormal))
            .collect();

        let z = na::DMatrix::from_vec(2, SAMPLE_SIZE, z_vec);
        let samples_target = na::DMatrix::from_row_slice(SAMPLE_SIZE, 2, (chol.l() * z).as_slice());

        // Run MCMC, discard burn-in
        let samples = mh.run(SAMPLE_SIZE / N_CHAINS + BURNIN, BURNIN).concat();

        let flattened: Vec<f64> = samples.into_iter().flatten().collect();
        let samples_keep = na::DMatrix::from_row_slice(SAMPLE_SIZE, 2, &flattened);

        // Validate mean & covariance
        let mean_mcmc = samples_keep.row_mean();
        let cov_mcmc = stats::cov(&samples_keep).unwrap();

        // Compare log probabilities via two-sample KS test
        let log_prob_mcmc: Vec<f64> = samples_keep
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
        let initial_state = vec![0.0, 0.0];
        let proposal = IsotropicGaussian::new(1.0).set_seed(seed);
        let mut mh =
            MetropolisHastings::new(target, proposal, initial_state, N_CHAINS).set_seed(seed);

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
        let samples = mh.run(SAMPLE_SIZE / N_CHAINS + BURNIN, BURNIN).concat();
        let flattened: Vec<f64> = samples.into_iter().flatten().collect();
        let samples_mcmc = na::DMatrix::from_row_slice(SAMPLE_SIZE, 2, &flattened);

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
}
