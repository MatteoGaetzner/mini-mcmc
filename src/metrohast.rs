//! An implementation of the Metropolis-Hastings sampler, built around a generic
//! target distribution `D` and proposal distribution `Q`.

use crate::distributions::{ProposalDistribution, TargetDistribution};
use rand::Rng;

/// The Metropolis-Hastings sampler, parameterized by:
/// - `D`: the target distribution,
/// - `Q`: the proposal distribution.
///
/// Maintains the current state of the chain, as well as an RNG.
pub struct MetropolisHastings<D, Q> {
    /// The target distribution we want to sample from.
    pub target: D,

    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,

    /// The current state of the Markov chain.
    pub current_state: Vec<f64>,

    /// RNG for generating random proposals and accept/reject decisions.
    pub rng: rand::rngs::ThreadRng,
}

impl<D, Q> MetropolisHastings<D, Q>
where
    D: TargetDistribution,
    Q: ProposalDistribution,
{
    /// Constructs a new Metropolis-Hastings sampler with a given target and proposal,
    /// initializing the chain at `initial_state`.
    pub fn new(target: D, proposal: Q, initial_state: Vec<f64>) -> Self {
        Self {
            target,
            proposal,
            current_state: initial_state,
            rng: rand::thread_rng(),
        }
    }

    /// Performs one Metropolis-Hastings update. Proposes a new state, computes the
    /// acceptance probability, and returns the (potentially updated) current state.
    ///
    /// # Returns
    ///
    /// The current state of the chain after this iteration.
    pub fn step(&mut self) -> Vec<f64> {
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
        let u: f64 = self.rng.gen();
        if log_accept_ratio > u.ln() {
            self.current_state = proposed;
        }
        self.current_state.clone()
    }
}
