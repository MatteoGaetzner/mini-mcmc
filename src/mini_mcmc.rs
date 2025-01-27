use rand::Rng;
use rand_distr::{Distribution, Normal};

pub trait TargetDistribution {
    fn log_prob(&self, theta: &[f64]) -> f64;
}

pub trait ProposalDistribution {
    /// Sample from q(theta'|theta)
    fn sample(&mut self, current: &[f64]) -> Vec<f64>;

    /// Log of q(theta'|theta)
    fn log_proposal(&self, from: &[f64], to: &[f64]) -> f64;
}

pub struct MetropolisHastings<D, Q> {
    pub target: D,
    pub proposal: Q,
    pub current_state: Vec<f64>,
    pub rng: rand::rngs::ThreadRng,
}

impl<D, Q> MetropolisHastings<D, Q>
where
    D: TargetDistribution,
    Q: ProposalDistribution,
{
    pub fn new(target: D, proposal: Q, initial_state: Vec<f64>) -> Self {
        Self {
            target,
            proposal,
            current_state: initial_state,
            rng: rand::thread_rng(),
        }
    }

    /// Perform one iteration of Metropolis-Hastings.
    pub fn step(&mut self) -> Vec<f64> {
        let current_lp = self.target.log_prob(&self.current_state);
        let proposed = self.proposal.sample(&self.current_state);
        let proposed_lp = self.target.log_prob(&proposed);

        let log_q_forward = self.proposal.log_proposal(&self.current_state, &proposed);
        let log_q_backward = self.proposal.log_proposal(&proposed, &self.current_state);

        // acceptance ratio in log space
        let log_accept_ratio = (proposed_lp + log_q_backward) - (current_lp + log_q_forward);

        let u: f64 = self.rng.gen(); // uniform in [0, 1)
        if log_accept_ratio.exp() > u {
            // accept
            self.current_state = proposed;
        }
        self.current_state.clone()
    }
}

pub struct Gaussian2D {
    pub mean: [f64; 2],
    pub cov: [[f64; 2]; 2], // for simplicity, assume diagonal or handle a general covariance
}

impl TargetDistribution for Gaussian2D {
    fn log_prob(&self, theta: &[f64]) -> f64 {
        let x = theta[0];
        let y = theta[1];
        // If diagonal covariance: var_x = cov[0][0], var_y = cov[1][1]
        let var_x = self.cov[0][0];
        let var_y = self.cov[1][1];
        // log prob ~ -(x - mean_x)^2 / (2 var_x) - (y - mean_y)^2 / (2 var_y)
        let log_px = -(x - self.mean[0]).powi(2) / (2.0 * var_x);
        let log_py = -(y - self.mean[1]).powi(2) / (2.0 * var_y);
        log_px + log_py
    }
}

pub struct GaussianProposal {
    pub std: f64,
    rng: rand::rngs::ThreadRng,
}

impl GaussianProposal {
    pub fn new(std: f64) -> Self {
        Self {
            std,
            rng: rand::thread_rng(),
        }
    }
}

impl ProposalDistribution for GaussianProposal {
    fn sample(&mut self, current: &[f64]) -> Vec<f64> {
        let mut proposal = Vec::with_capacity(current.len());
        for &val in current {
            let noise = Normal::new(0.0, self.std).unwrap().sample(&mut self.rng);
            proposal.push(val + noise);
        }
        proposal
    }

    fn log_proposal(&self, from: &[f64], to: &[f64]) -> f64 {
        // for Gaussian proposal, log_q(theta'|theta) = sum of log N((theta'-theta)/std)
        let mut lp = 0.0;
        for (&f, &t) in from.iter().zip(to.iter()) {
            let diff = t - f;
            let exponent = -(diff * diff) / (2.0 * self.std * self.std);
            let norm_constant = -(0.5 * (2.0 * std::f64::consts::PI * self.std * self.std).ln());
            lp += exponent + norm_constant;
        }
        lp
    }
}
