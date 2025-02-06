//! An implementation of the Metropolis-Hastings sampler, built around a generic
//! target distribution `D` and proposal distribution `Q`.

use rand::prelude::*;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

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
    pub chains: Vec<MarkovChain<D, Q>>,
}

#[derive(Clone)]
pub struct MarkovChain<D, Q> {
    /// The target distribution we want to sample from.
    pub target: D,

    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,

    /// The current state of the Markov chain.
    pub current_state: Vec<f64>,
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
            .map(|_| MarkovChain {
                target: target.clone(),
                proposal: proposal.clone(),
                current_state: initial_state.clone(),
            })
            .collect();

        Self {
            target,
            proposal,
            chains,
        }
    }

    pub fn run(&mut self, n_steps: usize, burnin: usize) -> Vec<Vec<Vec<f64>>> {
        let num_threads = match std::thread::available_parallelism() {
            Ok(v) => v.get(),
            Err(_) => {
                println!("Warning: determinining number of threads we should use failed. falling back to using a single thread.");
                1
            }
        };

        let res: Vec<ChainResult> = self
            .chains
            .par_iter_mut()
            .with_max_len(num_threads) // Limit concurrency
            .map(|chain| {
                let samples = chain.run(n_steps);
                (chain.current_state.clone(), samples)
            })
            .collect();

        for (i, (final_state, _)) in res.iter().enumerate() {
            self.chains[i].current_state.clone_from(final_state);
        }

        res.into_par_iter()
            .map(|(_, samples)| samples[burnin..].to_vec())
            .collect()
    }
}

impl<D, Q> MarkovChain<D, Q>
where
    D: TargetDistribution + Clone,
    Q: ProposalDistribution + Clone,
{
    pub fn run(&mut self, n_steps: usize) -> Vec<Vec<f64>> {
        println!("Starting to run a new chain with {n_steps} steps.");
        let mut out = Vec::new();
        let mut rng = rand::thread_rng();
        out.reserve(n_steps);
        out.extend(std::iter::repeat_with(|| self.step(&mut rng)).take(n_steps));
        println!("Finished running a chain.");
        out
    }

    /// Performs one Metropolis-Hastings update. Proposes a new state, computes the
    /// acceptance probability, and returns the (potentially updated) current state.
    ///
    /// # Returns
    ///
    /// The current state of the chain after this iteration.
    pub fn step(&mut self, rng: &mut ThreadRng) -> Vec<f64> {
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
    use rand_distr::StandardNormal;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use crate::{
        distributions::{Gaussian2D, IsotropicGaussian},
        ks_test::two_sample_ks_test,
        stats,
    };
    use nalgebra as na;

    #[test]
    fn four_chains_test() {
        const SAMPLE_SIZE: usize = 100_000;
        const BURNIN: usize = 10_000;
        const N_CHAINS: usize = 4;

        // Target distribution & sampler setup
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[4.0, 2.0], [2.0, 3.0]].into(),
        };
        let initial_state = vec![10.0, 12.0];
        let proposal = IsotropicGaussian::new(1.0);
        let mut mh = MetropolisHastings::new(target, proposal, initial_state, N_CHAINS);

        // Generate "true" samples from the target using Cholesky
        let chol = na::Cholesky::new(mh.chains[0].target.cov).expect("Cov not positive definite");
        let z_vec: Vec<f64> = (0..(2 * SAMPLE_SIZE) as i32)
            .into_par_iter()
            .map_init(rand::thread_rng, |rng, _| rng.sample(StandardNormal))
            .collect();
        let z = na::DMatrix::from_vec(2, SAMPLE_SIZE, z_vec);
        let samples_target = na::DMatrix::from_row_slice(SAMPLE_SIZE, 2, (chol.l() * z).as_slice());

        // Run MCMC, discard burn-in
        let mut samples = mh.run(SAMPLE_SIZE / N_CHAINS + BURNIN, BURNIN).concat();
        samples.shuffle(&mut SmallRng::from_entropy());

        let flattened: Vec<f64> = samples.into_iter().flatten().collect();
        let samples_keep = na::DMatrix::from_row_slice(SAMPLE_SIZE, 2, &flattened);

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
        let mean_mcmc = samples_keep.row_mean();
        let cov_mcmc = stats::cov(&samples_keep).unwrap();
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
}
