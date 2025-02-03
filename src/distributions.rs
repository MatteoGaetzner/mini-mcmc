//! Defines target and proposal distributions for 2D and arbitrary-dimensional Gaussian models,
//! along with traits for sampling and evaluating log-probabilities.

use nalgebra::{Matrix2, Vector2};
use rand_distr::num_traits::ToPrimitive;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// A trait for generating proposals in Metropolis-Hastings or similar algorithms.
pub trait ProposalDistribution {
    /// Sajples a new point from q(θ′ | θ).
    fn sample(&mut self, current: &[f64]) -> Vec<f64>;

    /// Evaluates log q(θ′ | θ).
    fn log_prob(&self, from: &[f64], to: &[f64]) -> f64;
}

/// A trait for distributions from which we want to sample via some MCMC method.
/// The `unnorm_log_prob` is the log of the unnormalized density.
pub trait TargetDistribution {
    fn unnorm_log_prob(&self, theta: &[f64]) -> f64;
}

/// A trait for distributions that provide a normalized log-density (e.g., for diagnostics).
#[allow(dead_code)]
pub trait Normalized {
    fn log_prob(&self, theta: &[f64]) -> f64;
}

/// A 2D Gaussian distribution parameterized by mean and a 2×2 covariance matrix.
pub struct Gaussian2D {
    pub mean: Vector2<f64>,
    pub cov: Matrix2<f64>,
}

impl Normalized for Gaussian2D {
    /// Computes the fully normalized log-density of a 2D Gaussian.
    fn log_prob(&self, theta: &[f64]) -> f64 {
        let term_1 = -(2.0 * PI).ln();
        let (a, b, c, d) = (
            self.cov[(0, 0)],
            self.cov[(0, 1)],
            self.cov[(1, 0)],
            self.cov[(1, 1)],
        );
        let det = a * d - b * c;
        let term_2 = -0.5 * det.abs().ln();

        let x = Vector2::new(theta[0], theta[1]);
        let diff = x - self.mean;
        let inv_cov = Matrix2::new(d, -b, -c, a) / det;
        let term_3 = -0.5 * (diff.transpose() * inv_cov * diff)[(0, 0)];

        term_1 + term_2 + term_3
    }
}

impl TargetDistribution for Gaussian2D {
    /// Computes the unnormalized log-density of a 2D Gaussian.
    fn unnorm_log_prob(&self, theta: &[f64]) -> f64 {
        let (a, b, c, d) = (
            self.cov[(0, 0)],
            self.cov[(0, 1)],
            self.cov[(1, 0)],
            self.cov[(1, 1)],
        );
        let det = a * d - b * c;
        let x = Vector2::new(theta[0], theta[1]);
        let diff = x - self.mean;
        let inv_cov = Matrix2::new(d, -b, -c, a) / det;
        -0.5 * (diff.transpose() * inv_cov * diff)[(0, 0)]
    }
}

pub struct IsotropicGaussian {
    pub std: f64,
    rng: rand::rngs::ThreadRng,
}

impl IsotropicGaussian {
    pub fn new(std: f64) -> Self {
        Self {
            std,
            rng: rand::thread_rng(),
        }
    }
}

impl ProposalDistribution for IsotropicGaussian {
    fn sample(&mut self, current: &[f64]) -> Vec<f64> {
        let mut proposal = Vec::with_capacity(current.len());
        for &val in current {
            let noise = Normal::new(0.0, self.std).unwrap().sample(&mut self.rng);
            proposal.push(val + noise);
        }
        proposal
    }

    fn log_prob(&self, from: &[f64], to: &[f64]) -> f64 {
        // for Gaussian proposal, log_q(theta'|theta) = sum of log N((theta'-theta)/std)
        let mut lp = 0.0;
        for (&f, &t) in from.iter().zip(to.iter()) {
            let diff = t - f;
            let exponent = -(diff * diff) / (2.0 * self.std * self.std);
            lp += exponent;
        }
        lp += -from.len().to_f64().unwrap() * 0.5 * (2.0 * PI * self.std * self.std).ln();
        lp
    }
}

impl TargetDistribution for IsotropicGaussian {
    fn unnorm_log_prob(&self, theta: &[f64]) -> f64 {
        theta.iter().fold(0., |sum, x| sum - 0.5 * (x.powf(2.)))
    }
}
