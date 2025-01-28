use nalgebra::{Matrix2, Vector2};
use rand::Rng;
use rand_distr::num_traits::ToPrimitive;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

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
        let proposed = self.proposal.sample(&self.current_state);

        let current_lp = self.target.log_prob(&self.current_state); // log p(x)
        let proposed_lp = self.target.log_prob(&proposed); // log p(x')

        // log p(x' | x), log p(x | x')
        let log_q_forward = self.proposal.log_proposal(&self.current_state, &proposed);
        let log_q_backward = self.proposal.log_proposal(&proposed, &self.current_state);

        // acceptance ratio in log space
        // log p(x') + log p(x | x') - (log p(x) + log p(x' | x))
        let log_accept_ratio = (proposed_lp + log_q_backward) - (current_lp + log_q_forward);

        let u: f64 = self.rng.gen(); // uniform in [0, 1)
        if log_accept_ratio > u.ln() {
            self.current_state = proposed;
        }
        self.current_state.clone()
    }
}

pub struct Gaussian2D {
    pub mean: Vector2<f64>,
    pub cov: Matrix2<f64>,
}

impl TargetDistribution for Gaussian2D {
    fn log_prob(&self, theta: &[f64]) -> f64 {
        // - log(2 pi) - 0.5 log |det(Sigma)|- 0.5 (x - mu)^T Sigma^{-1} (x - mu)
        // = -log (2 pi)                                                        (term 1)
        //   - 0.5 log |Sigma_11 * Sigma_22 - Sigma_12 * Sigma_12|              (term 2)
        //   - 0.5 (x - mu)^T Sigma^{-1} (x - mu)                               (term 3)

        // Compute term 1
        // println!("mean: {}", self.mean);
        // println!("cov: {}", self.cov);

        let term_1 = -(2.0 * PI).ln();
        // println!("term_1: {}", term_1);

        // Compute term 2
        let (a, b, c, d) = (
            self.cov[(0, 0)],
            self.cov[(0, 1)],
            self.cov[(1, 0)],
            self.cov[(1, 1)],
        );
        let det = a * d - b * c;
        let term_2 = -0.5 * det.abs().ln();
        // println!("term_2: {}", term_2);

        // Compute term 3
        let x = Vector2::new(theta[0], theta[1]);
        let diff = x - self.mean;
        let inv_cov = Matrix2::new(d, -b, -c, a) / det;
        // println!("diff: {}", diff);
        // println!("inv_cov: {}", inv_cov);
        let term_3 = -0.5 * (diff.transpose() * inv_cov * diff)[(0, 0)];
        // println!("term_3: {}", term_3);

        // Combine terms
        let log_prob = term_1 + term_2 + term_3;

        // println!("log_prob: {}", log_prob);
        log_prob
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
            lp += exponent;
            // println!("  lp: {}", lp);
        }
        lp += -from.len().to_f64().unwrap() * 0.5 * (2.0 * PI * self.std * self.std).ln();
        // println!("  lp: {}", lp);
        lp
    }
}
