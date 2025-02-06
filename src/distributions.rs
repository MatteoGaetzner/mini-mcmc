//! Defines target and proposal distributions for 2D and arbitrary-dimensional Gaussian models,
//! along with traits for sampling and evaluating log-probabilities.

use nalgebra::{Matrix2, Vector2};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// A trait for generating proposals in Metropolis-Hastings or similar algorithms.
pub trait ProposalDistribution {
    /// Sajples a new point from q(θ′ | θ).
    fn sample(&mut self, current: &[f64]) -> Vec<f64>;

    /// Evaluates log q(θ′ | θ).
    fn log_prob(&self, from: &[f64], to: &[f64]) -> f64;

    /// Set random seed.
    fn set_seed(self, seed: u64) -> Self;
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
#[derive(Clone)]
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

#[derive(Clone)]
pub struct IsotropicGaussian {
    pub std: f64,
    rng: SmallRng,
}

impl IsotropicGaussian {
    pub fn new(std: f64) -> Self {
        Self {
            std,
            rng: SmallRng::from_entropy(),
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

    fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }
}

impl TargetDistribution for IsotropicGaussian {
    fn unnorm_log_prob(&self, theta: &[f64]) -> f64 {
        -0.5 * theta.iter().map(|x| x * x).sum::<f64>() / (self.std * self.std)
    }
}

fn _normalize_isogauss(x: f64, d: usize, std: f64) -> f64 {
    let log_normalizer = -((d as f64) / 2.0) * ((2.0_f64).ln() + PI.ln() + 2.0 * std.ln());
    (x + log_normalizer).exp()
}

// impl TargetDistribution for IsotropicGaussian {
//     fn unnorm_log_prob(&self, theta: &[f64]) -> f64 {
//         -0.5 * theta.iter().fold(0., |sum, x| sum + (x * x)) / (self.std * self.std)
//     }
// }
//
// fn _normalize_isogauss(x: f64, d: usize, std: f64) -> f64 {
//     (1.0 / ((2.0 * PI).sqrt() * std).powi(d as i32)) * x.exp()
// }

#[test]
fn iso_gauss_unnorm_log_prob_test_1() {
    let distr = IsotropicGaussian::new(1.0);
    let p = _normalize_isogauss(distr.unnorm_log_prob(&[1.0]), 1, distr.std);
    let true_p = 0.24197072451914337;
    let diff = (p - true_p).abs();
    assert!(
        diff < 1e-7,
        "Expected diff < 1e-7, got {diff} with p={p} (expected ~{true_p})."
    );
}

#[test]
fn iso_gauss_unnorm_log_prob_test_2() {
    let distr = IsotropicGaussian::new(2.0);
    let p = _normalize_isogauss(distr.unnorm_log_prob(&[0.42, 9.6]), 2, distr.std);
    let true_p = 3.864661987252467e-7;
    let diff = (p - true_p).abs();
    assert!(
        diff < 1e-15,
        "Expected diff < 1e-15, got {diff} with p={p} (expected ~{true_p})"
    );
}

#[test]
fn iso_gauss_unnorm_log_prob_test_3() {
    let distr = IsotropicGaussian::new(3.0);
    let p = _normalize_isogauss(distr.unnorm_log_prob(&[1.0, 2.0, 3.0]), 3, distr.std);
    let true_p = 0.001080393185560214;
    let diff = (p - true_p).abs();
    assert!(
        diff < 1e-8,
        "Expected diff < 1e-8, got {diff} with p={p} (expected ~{true_p})"
    );
}
