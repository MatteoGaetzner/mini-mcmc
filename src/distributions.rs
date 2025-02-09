/*!
Defines target and proposal distributions for 2D (and by extension, arbitrary-dimensional)
Gaussian models along with traits for sampling and evaluating log-probabilities. It also
defines a simple discrete distribution (a categorical distribution).

This module is generic over the floating-point precision (e.g., `f32` or `f64`) using
the [`num_traits::Float`] trait, and it provides a new trait for discrete distributions.

# Examples

### Continuous Distributions

```rust
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian, ProposalDistribution, TargetDistribution, Normalized};
use nalgebra::{Vector2, Matrix2};

// Create a 2D Gaussian target distribution using f64.
let mean = Vector2::new(0.0, 0.0);
let cov = Matrix2::new(1.0, 0.0, 0.0, 1.0);
let gauss: Gaussian2D<f64> = Gaussian2D { mean, cov };
let logp = gauss.log_prob(&vec![0.5, -0.5]);
println!("Normalized log-probability: {}", logp);

// Create an isotropic Gaussian proposal distribution.
let mut proposal: IsotropicGaussian<f64> = IsotropicGaussian::new(1.0);
let current = vec![0.0, 0.0];
let candidate = proposal.sample(&current);
println!("Candidate state: {:?}", candidate);
```

### Discrete Distributions

```rust
use mini_mcmc::distributions::{Categorical, DiscreteDistribution};

// Create a categorical distribution over three categories.
let mut cat = Categorical::new(vec![0.2f64, 0.3, 0.5]);
let sample = cat.sample();
println!("Sampled category: {}", sample);
let logp = cat.log_prob(sample);
println!("Log-probability of sampled category: {}", logp);
```
*/

use nalgebra::{Matrix2, Vector2};
use num_traits::Float;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::num_traits::ToPrimitive;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use std::ops::AddAssign;

/// A trait for generating proposals in Metropolis–Hastings or similar algorithms.
/// The state type `S` is typically a vector of continuous values.
pub trait ProposalDistribution<S, T: Float> {
    /// Samples a new point from q(x' | x).
    fn sample(&mut self, current: &S) -> S;

    /// Evaluates log q(x' | x).
    fn log_prob(&self, from: &S, to: &S) -> T;

    /// Returns a new instance of this proposal distribution seeded with `seed`.
    fn set_seed(self, seed: u64) -> Self;
}

/// A trait for continuous target distributions from which we want to sample.
/// The state type `S` is typically a vector of continuous values.
pub trait TargetDistribution<S, T: Float> {
    /// Returns the log of the unnormalized density for state `theta`.
    fn unnorm_log_prob(&self, theta: &S) -> T;
}

/// A trait for distributions that provide a normalized log-density (e.g., for diagnostics).
pub trait Normalized<S, T: Float> {
    /// Returns the normalized log-density for state `theta`.
    fn log_prob(&self, theta: &S) -> T;
}

/// A trait for discrete distributions whose state is represented as an index.
pub trait DiscreteDistribution {
    /// Samples an index from the distribution.
    fn sample(&mut self) -> usize;
    /// Evaluates the log-probability of the given index.
    fn log_prob(&self, index: usize) -> f64;
}

/**
A 2D Gaussian distribution parameterized by a mean vector and a 2×2 covariance matrix.

This struct is generic over the floating-point type `T` (e.g. `f32` or `f64`).

# Examples

```rust
use mini_mcmc::distributions::{Gaussian2D, Normalized};
use nalgebra::{Vector2, Matrix2};

let mean = Vector2::new(0.0, 0.0);
let cov = Matrix2::new(1.0, 0.0, 0.0, 1.0);
let gauss: Gaussian2D<f64> = Gaussian2D { mean, cov };
let lp = gauss.log_prob(&vec![0.5, -0.5]);
println!("Log probability: {}", lp);
```
*/
#[derive(Clone, Copy)]
pub struct Gaussian2D<T: Float + ToPrimitive> {
    pub mean: Vector2<T>,
    pub cov: Matrix2<T>,
}

impl<T: Float> Normalized<Vec<T>, T> for Gaussian2D<T>
where
    T: Float
        + ToPrimitive
        + std::ops::SubAssign
        + std::fmt::Debug
        + AddAssign
        + std::ops::DivAssign
        + std::ops::MulAssign
        + 'static,
{
    /// Computes the fully normalized log-density of a 2D Gaussian.
    fn log_prob(&self, theta: &Vec<T>) -> T {
        let term_1 = -(T::from(2.0).unwrap() * T::from(PI).unwrap()).ln();
        let (a, b, c, d) = (
            self.cov[(0, 0)],
            self.cov[(0, 1)],
            self.cov[(1, 0)],
            self.cov[(1, 1)],
        );
        let det = a * d - b * c;
        let half = T::from(0.5).unwrap();
        let term_2 = -half * det.abs().ln();

        let x = Vector2::new(theta[0], theta[1]);
        let diff = x - self.mean;
        let inv_cov = Matrix2::new(d, -b, -c, a) / det;
        let term_3 = -half * (diff.transpose() * inv_cov * diff)[(0, 0)];
        term_1 + term_2 + term_3
    }
}

impl<T> TargetDistribution<Vec<T>, T> for Gaussian2D<T>
where
    T: Float
        + ToPrimitive
        + std::fmt::Debug
        + std::ops::SubAssign
        + std::ops::DivAssign
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::Neg
        + 'static,
{
    fn unnorm_log_prob(&self, theta: &Vec<T>) -> T {
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
        -T::from(0.5).unwrap() * (diff.transpose() * inv_cov * diff)[(0, 0)]
    }
}

/**
An isotropic Gaussian distribution used as a proposal distribution.

This distribution adds independent Gaussian noise (with mean 0 and standard deviation `std`)
to each coordinate of the current state. It is generic over the floating-point type `T`.

# Examples

```rust
use mini_mcmc::distributions::{IsotropicGaussian, ProposalDistribution};

let mut proposal: IsotropicGaussian<f64> = IsotropicGaussian::new(1.0);
let current = vec![0.0, 0.0];
let candidate = proposal.sample(&current);
println!("Candidate state: {:?}", candidate);
```
*/
#[derive(Clone)]
pub struct IsotropicGaussian<T: Float + ToPrimitive> {
    pub std: T,
    rng: SmallRng,
}

impl<T: Float + ToPrimitive> IsotropicGaussian<T> {
    /// Creates a new isotropic Gaussian proposal distribution with the specified standard deviation.
    pub fn new(std: T) -> Self {
        Self {
            std,
            rng: SmallRng::from_entropy(),
        }
    }
}

impl<T: Float + std::ops::AddAssign> ProposalDistribution<Vec<T>, T> for IsotropicGaussian<T>
where
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
{
    fn sample(&mut self, current: &Vec<T>) -> Vec<T> {
        let normal = Normal::new(T::zero(), self.std)
            .expect("Expecting creation of normal distribution to succeed.");
        normal
            .sample_iter(&mut self.rng)
            .zip(current)
            .map(|(x, eps)| x + *eps)
            .collect()
    }

    fn log_prob(&self, from: &Vec<T>, to: &Vec<T>) -> T {
        let mut lp = T::zero();
        let d = T::from(from.len()).unwrap();
        let two = T::from(2).unwrap();
        let var = self.std * self.std;
        for (&f, &t) in from.iter().zip(to.iter()) {
            let diff = t - f;
            let exponent = -(diff * diff) / (two * var);
            lp += exponent;
        }
        lp += -d * T::from(0.5).unwrap() * (var * T::from(PI).unwrap() * self.std * self.std).ln();
        lp
    }

    fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }
}

impl<T: Float> TargetDistribution<Vec<T>, T> for IsotropicGaussian<T> {
    fn unnorm_log_prob(&self, theta: &Vec<T>) -> T {
        let mut sum = T::zero();
        for &x in theta.iter() {
            sum = sum + x * x
        }
        -T::from(0.5).unwrap() * sum / (self.std * self.std)
    }
}

/**
A categorical distribution represents a discrete probability distribution over a finite set of categories.

The probabilities in `probs` should sum to 1 (or they will be normalized automatically).

# Examples

```rust
use mini_mcmc::distributions::{Categorical, DiscreteDistribution};

let mut cat = Categorical::new(vec![0.2f64, 0.3, 0.5]);
let sample = cat.sample();
println!("Sampled category: {}", sample);
let logp = cat.log_prob(sample);
println!("Log probability of category {}: {}", sample, logp);
```
*/
#[derive(Clone)]
pub struct Categorical<T: Float + ToPrimitive> {
    pub probs: Vec<T>,
    rng: SmallRng,
}

impl<T: Float + ToPrimitive> Categorical<T> {
    /// Creates a new categorical distribution from a vector of probabilities.
    /// The probabilities will be normalized so that they sum to 1.
    pub fn new(probs: Vec<T>) -> Self {
        let sum: T = probs.iter().cloned().fold(T::zero(), |acc, x| acc + x);
        let normalized: Vec<T> = probs.into_iter().map(|p| p / sum).collect();
        Self {
            probs: normalized,
            rng: SmallRng::from_entropy(),
        }
    }
}

impl<T: Float + ToPrimitive> DiscreteDistribution for Categorical<T> {
    fn sample(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        let mut cum = 0.0;
        for (i, &p) in self.probs.iter().enumerate() {
            cum += p.to_f64().unwrap();
            if r < cum {
                return i;
            }
        }
        self.probs.len() - 1
    }

    fn log_prob(&self, index: usize) -> f64 {
        if index < self.probs.len() {
            self.probs[index].to_f64().unwrap().ln()
        } else {
            f64::NEG_INFINITY
        }
    }
}

#[cfg(test)]
mod distributions_tests {
    use super::*;

    /**
    A helper function to normalize the unnormalized log probability of an isotropic Gaussian
    into a proper probability value (by applying the appropriate constant).

    # Arguments

    * `x` - The unnormalized log probability.
    * `d` - The dimensionality of the state.
    * `std` - The standard deviation used in the isotropic Gaussian.

    # Returns

    Returns the normalized probability as an `f64`.
    */
    fn normalize_isogauss(x: f64, d: usize, std: f64) -> f64 {
        let log_normalizer = -((d as f64) / 2.0) * ((2.0_f64).ln() + PI.ln() + 2.0 * std.ln());
        (x + log_normalizer).exp()
    }

    #[test]
    fn iso_gauss_unnorm_log_prob_test_1() {
        let distr = IsotropicGaussian::new(1.0);
        let p = normalize_isogauss(distr.unnorm_log_prob(&[1.0].to_vec()), 1, distr.std);
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
        let p = normalize_isogauss(distr.unnorm_log_prob(&[0.42, 9.6].to_vec()), 2, distr.std);
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
        let p = normalize_isogauss(
            distr.unnorm_log_prob(&[1.0, 2.0, 3.0].to_vec()),
            3,
            distr.std,
        );
        let true_p = 0.001080393185560214;
        let diff = (p - true_p).abs();
        assert!(
            diff < 1e-8,
            "Expected diff < 1e-8, got {diff} with p={p} (expected ~{true_p})"
        );
    }
}
