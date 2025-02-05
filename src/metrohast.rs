//! An implementation of the Metropolis-Hastings sampler, built around a generic
//! target distribution `D` and proposal distribution `Q`.

use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::thread;

use crate::distributions::{ProposalDistribution, TargetDistribution};

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
        let res: Vec<(Vec<f64>, Vec<Vec<f64>>)> = thread::scope(|s| {
            self.chains
                .clone()
                .into_iter()
                .map(|mut chain| s.spawn(move || (chain.current_state.clone(), chain.run(n_steps))))
                .map(|handle| handle.join().expect("Expected joining threads to succeed."))
                .collect()
        });

        for (i, (final_state, _)) in res.iter().enumerate() {
            self.chains[i].current_state = final_state.to_vec();
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
        let mut out = Vec::new();
        let mut rng = rand::thread_rng();
        out.reserve(n_steps);
        out.extend(std::iter::repeat_with(|| self.step(&mut rng)).take(n_steps));
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
