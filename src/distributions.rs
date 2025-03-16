/*!
Traits for defining continuous and discrete proposal- and target distributions.
Includes an implementations of commonly used distributions.

This module is generic over the floating-point precision (e.g. `f32` or `f64`)
using [`num_traits::Float`]. It also defines several traits:
- [`Target`] for densities we want to sample from,
- [`Proposal`] for proposal mechanisms,
- [`Normalized`] for distributions that can compute a fully normalized log-prob,
- [`Discrete`] for distributions over finite sets.

# Examples

### Continuous Distributions

```rust
use mini_mcmc::distributions::{
    Gaussian2D, IsotropicGaussian, Proposal,
    Target, Normalized
};
use ndarray::{arr1, arr2};

// ----------------------
// Example: Gaussian2D (2D with full covariance)
// ----------------------
let mean = arr1(&[0.0, 0.0]);
let cov = arr2(&[[1.0, 0.0],
                [0.0, 1.0]]);
let gauss: Gaussian2D<f64> = Gaussian2D { mean, cov };

// Compute the fully normalized log-prob at (0.5, -0.5):
let logp = gauss.log_prob(&vec![0.5, -0.5]);
println!("Normalized log-probability (2D Gaussian): {}", logp);

// ----------------------
// Example: IsotropicGaussian (any dimension)
// ----------------------
let mut proposal: IsotropicGaussian<f64> = IsotropicGaussian::new(1.0);
let current = vec![0.0, 0.0];  // dimension = 2 in this example
let candidate = proposal.sample(&current);
println!("Candidate state: {:?}", candidate);
*/

use ndarray::{arr1, arr2, Array1, Array2, NdFloat};
use num_traits::Float;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use std::ops::AddAssign;

/// A trait for generating proposals Metropolis–Hastings-like algorithms.
/// The state type `T` is typically a vector of continuous values.
pub trait Proposal<T, F: Float> {
    /// Samples a new point from q(x' | x).
    fn sample(&mut self, current: &[T]) -> Vec<T>;

    /// Evaluates log q(x' | x).
    fn log_prob(&self, from: &[T], to: &[T]) -> F;

    /// Returns a new instance of this proposal distribution seeded with `seed`.
    fn set_seed(self, seed: u64) -> Self;
}

/// A trait for continuous target distributions from which we want to sample.
/// The state type `T` is typically a vector of continuous values.
pub trait Target<T, F: Float> {
    /// Returns the log of the unnormalized density for state `theta`.
    fn unnorm_log_prob(&self, theta: &[T]) -> F;
}

/// A trait for distributions that provide a normalized log-density (e.g. for diagnostics).
pub trait Normalized<T, F: Float> {
    /// Returns the normalized log-density for state `theta`.
    fn log_prob(&self, theta: &[T]) -> F;
}

/** A trait for discrete distributions whose state is represented as an index.
 ```rust
 use mini_mcmc::distributions::{Categorical, Discrete};

 // Create a categorical distribution over three categories.
 let mut cat = Categorical::new(vec![0.2f64, 0.3, 0.5]);
 let sample = cat.sample();
 println!("Sampled category: {}", sample); // E.g. 1usize

 let logp = cat.log_prob(sample);
 println!("Log-probability of sampled category: {}", logp); // E.g. 0.3f64
```
*/
pub trait Discrete<T: Float> {
    /// Samples an index from the distribution.
    fn sample(&mut self) -> usize;
    /// Evaluates the log-probability of the given index.
    fn log_prob(&self, index: usize) -> T;
}

/**
A 2D Gaussian distribution parameterized by a mean vector and a 2×2 covariance matrix.

- The generic type `T` is typically `f32` or `f64`.
- Implements both [`Target`] (for unnormalized log-prob) and
  [`Normalized`] (for fully normalized log-prob).

# Example

```rust
use mini_mcmc::distributions::{Gaussian2D, Normalized};
use ndarray::{arr1, arr2};

let mean = arr1(&[0.0, 0.0]);
let cov = arr2(&[[1.0, 0.0],
                [0.0, 1.0]]);
let gauss: Gaussian2D<f64> = Gaussian2D { mean, cov };

let lp = gauss.log_prob(&vec![0.5, -0.5]);
println!("Normalized log probability: {}", lp);
```
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gaussian2D<T: Float> {
    pub mean: Array1<T>,
    pub cov: Array2<T>,
}

impl<T> Normalized<T, T> for Gaussian2D<T>
where
    T: NdFloat,
{
    /// Computes the fully normalized log-density of a 2D Gaussian.
    fn log_prob(&self, theta: &[T]) -> T {
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

        let x = arr1(theta);
        let diff = x - self.mean.clone();
        let inv_cov = arr2(&[[d, -b], [-c, a]]) / det;
        let term_3 = -half * diff.dot(&inv_cov).dot(&diff);
        term_1 + term_2 + term_3
    }
}

impl<T> Target<T, T> for Gaussian2D<T>
where
    T: NdFloat,
{
    fn unnorm_log_prob(&self, theta: &[T]) -> T {
        let (a, b, c, d) = (
            self.cov[(0, 0)],
            self.cov[(0, 1)],
            self.cov[(1, 0)],
            self.cov[(1, 1)],
        );
        let det = a * d - b * c;
        let x = arr1(theta);
        let diff = x - self.mean.clone();
        let inv_cov = arr2(&[[d, -b], [-c, a]]) / det;
        -T::from(0.5).unwrap() * diff.dot(&inv_cov).dot(&diff)
    }
}

/**
An *isotropic* Gaussian distribution usable as either a target or a proposal
in MCMC. It works for **any dimension** because it applies independent
Gaussian noise (`mean = 0`, `std = self.std`) to each coordinate.

- Implements [`Proposal`] so it can propose new states
  from a current state.
- Also implements [`Target`] for an unnormalized log-prob,
  which might be useful if you want to treat it as a target distribution
  in simplified scenarios.

# Examples

```rust
use mini_mcmc::distributions::{IsotropicGaussian, Proposal};

let mut proposal: IsotropicGaussian<f64> = IsotropicGaussian::new(1.0);
let current = vec![0.0, 0.0, 0.0]; // dimension = 3
let candidate = proposal.sample(&current);
println!("Candidate state: {:?}", candidate);

// Evaluate log q(candidate | current):
let logq = proposal.log_prob(&current, &candidate);
println!("Log of the proposal density: {}", logq);
```
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsotropicGaussian<T: Float> {
    pub std: T,
    rng: SmallRng,
}

impl<T: Float> IsotropicGaussian<T> {
    /// Creates a new isotropic Gaussian proposal distribution with the specified standard deviation.
    pub fn new(std: T) -> Self {
        Self {
            std,
            rng: SmallRng::from_entropy(),
        }
    }
}

impl<T: Float + std::ops::AddAssign> Proposal<T, T> for IsotropicGaussian<T>
where
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
{
    fn sample(&mut self, current: &[T]) -> Vec<T> {
        let normal = Normal::new(T::zero(), self.std)
            .expect("Expecting creation of normal distribution to succeed.");
        normal
            .sample_iter(&mut self.rng)
            .zip(current)
            .map(|(x, eps)| x + *eps)
            .collect()
    }

    fn log_prob(&self, from: &[T], to: &[T]) -> T {
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

impl<T: Float> Target<T, T> for IsotropicGaussian<T> {
    fn unnorm_log_prob(&self, theta: &[T]) -> T {
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
use mini_mcmc::distributions::{Categorical, Discrete};

let mut cat = Categorical::new(vec![0.2f64, 0.3, 0.5]);
let sample = cat.sample();
println!("Sampled category: {}", sample);
let logp = cat.log_prob(sample);
println!("Log probability of category {}: {}", sample, logp);
```
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Categorical<T>
where
    T: Float + std::ops::AddAssign,
{
    pub probs: Vec<T>,
    rng: SmallRng,
}

impl<T: Float + std::ops::AddAssign> Categorical<T> {
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

impl<T: Float + std::ops::AddAssign> Discrete<T> for Categorical<T>
where
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    fn sample(&mut self) -> usize {
        let r: T = self.rng.gen();
        let mut cum: T = T::zero();
        let mut k = self.probs.len() - 1;
        for (i, &p) in self.probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                k = i;
                break;
            }
        }
        k
    }

    fn log_prob(&self, index: usize) -> T {
        if index < self.probs.len() {
            self.probs[index].ln()
        } else {
            T::neg_infinity()
        }
    }
}

impl<T: Float + AddAssign> Target<usize, T> for Categorical<T>
where
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    fn unnorm_log_prob(&self, theta: &[usize]) -> T {
        <Self as Discrete<T>>::log_prob(self, theta[0])
    }
}

/**
A trait for conditional distributions.

This trait specifies how to sample a single coordinate of a state given the entire current state.
It is primarily used in Gibbs sampling to update one coordinate at a time.
*/
pub trait Conditional<S> {
    fn sample(&mut self, index: usize, given: &[S]) -> S;
}

#[cfg(test)]
mod continuous_tests {
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
        let p = normalize_isogauss(distr.unnorm_log_prob(&[1.0]), 1, distr.std);
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
        let p = normalize_isogauss(distr.unnorm_log_prob(&[0.42, 9.6]), 2, distr.std);
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
        let p = normalize_isogauss(distr.unnorm_log_prob(&[1.0, 2.0, 3.0]), 3, distr.std);
        let true_p = 0.001080393185560214;
        let diff = (p - true_p).abs();
        assert!(
            diff < 1e-8,
            "Expected diff < 1e-8, got {diff} with p={p} (expected ~{true_p})"
        );
    }
}

#[cfg(test)]
mod categorical_tests {
    use super::*;

    /// A helper function to compare floating-point values with a given tolerance.
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ------------------------------------------------------
    // 1) Test log_prob correctness for f64
    // ------------------------------------------------------
    #[test]
    fn test_categorical_log_prob_f64() {
        let probs = vec![0.2, 0.3, 0.5];
        let cat = Categorical::<f64>::new(probs.clone());

        // Check log probabilities for each index
        let log_prob_0 = cat.log_prob(0);
        let log_prob_1 = cat.log_prob(1);
        let log_prob_2 = cat.log_prob(2);

        // Expected values
        let expected_0 = 0.2_f64.ln();
        let expected_1 = 0.3_f64.ln();
        let expected_2 = 0.5_f64.ln();

        let tol = 1e-7;
        assert!(
            approx_eq(log_prob_0, expected_0, tol),
            "Log prob mismatch at index 0: got {}, expected {}",
            log_prob_0,
            expected_0
        );
        assert!(
            approx_eq(log_prob_1, expected_1, tol),
            "Log prob mismatch at index 1: got {}, expected {}",
            log_prob_1,
            expected_1
        );
        assert!(
            approx_eq(log_prob_2, expected_2, tol),
            "Log prob mismatch at index 2: got {}, expected {}",
            log_prob_2,
            expected_2
        );

        // Out-of-bounds index should be NEG_INFINITY
        let log_prob_out = cat.log_prob(3);
        assert_eq!(
            log_prob_out,
            f64::NEG_INFINITY,
            "Out-of-bounds index did not return NEG_INFINITY"
        );
    }

    // ------------------------------------------------------
    // 2) Test sampling frequencies for f64
    // ------------------------------------------------------
    #[test]
    fn test_categorical_sampling_f64() {
        let probs = vec![0.2, 0.3, 0.5];
        let mut cat = Categorical::<f64>::new(probs.clone());

        let num_samples = 100_000;
        let mut counts = vec![0_usize; probs.len()];

        // Draw samples and tally outcomes
        for _ in 0..num_samples {
            let sample = cat.sample();
            counts[sample] += 1;
        }

        // Check empirical frequencies
        let tol = 0.01; // 1% absolute tolerance
        for (i, &count) in counts.iter().enumerate() {
            let freq = count as f64 / num_samples as f64;
            let expected = probs[i];
            assert!(
                approx_eq(freq, expected, tol),
                "Empirical freq for index {} is off: got {:.3}, expected {:.3}",
                i,
                freq,
                expected
            );
        }
    }

    // ------------------------------------------------------
    // 3) Test log_prob correctness for f32
    // ------------------------------------------------------
    #[test]
    fn test_categorical_log_prob_f32() {
        let probs = vec![0.1_f32, 0.4, 0.5];
        let cat = Categorical::<f32>::new(probs.clone());

        let log_prob_0: f32 = cat.log_prob(0);
        let log_prob_1 = cat.log_prob(1);
        let log_prob_2 = cat.log_prob(2);

        // For comparison, cast to f64
        let expected_0 = (0.1_f64).ln();
        let expected_1 = (0.4_f64).ln();
        let expected_2 = (0.5_f64).ln();

        let tol = 1e-6;
        assert!(
            approx_eq(log_prob_0.into(), expected_0, tol),
            "Log prob mismatch at index 0 (f32 -> f64 cast)"
        );
        assert!(
            approx_eq(log_prob_1.into(), expected_1, tol),
            "Log prob mismatch at index 1"
        );
        assert!(
            approx_eq(log_prob_2.into(), expected_2, tol),
            "Log prob mismatch at index 2"
        );

        // Out-of-bounds
        let log_prob_out = cat.log_prob(3);
        assert_eq!(log_prob_out, f32::NEG_INFINITY);
    }

    // ------------------------------------------------------
    // 4) Test sampling frequencies for f32
    // ------------------------------------------------------
    #[test]
    fn test_categorical_sampling_f32() {
        let probs = vec![0.1_f32, 0.4, 0.5];
        let mut cat = Categorical::<f32>::new(probs.clone());

        let num_samples = 100_000;
        let mut counts = vec![0_usize; probs.len()];

        for _ in 0..num_samples {
            let sample = cat.sample();
            counts[sample] += 1;
        }

        // Compare frequencies with expected probabilities
        let tol = 0.02; // might relax tolerance for f32
        for (i, &count) in counts.iter().enumerate() {
            let freq = count as f32 / num_samples as f32;
            let expected = probs[i];
            assert!(
                (freq - expected).abs() < tol,
                "Empirical freq for index {} is off: got {:.3}, expected {:.3}",
                i,
                freq,
                expected
            );
        }
    }

    #[test]
    fn test_categorical_sample_single_value() {
        let mut cat = Categorical {
            probs: vec![1.0_f64],
            rng: rand::rngs::SmallRng::from_seed(Default::default()),
        };

        let sampled_index = cat.sample();

        assert_eq!(
            sampled_index, 0,
            "Should return the last index (0) for a single-element vector"
        );
    }

    #[test]
    fn test_target_for_categorical_in_range() {
        // Create a categorical distribution with known probabilities.
        let probs = vec![0.2_f64, 0.3, 0.5];
        let cat = Categorical::new(probs.clone());
        // Call unnorm_log_prob with a valid index (say, index 1).
        let logp = cat.unnorm_log_prob(&[1]);
        // The expected log probability is ln(0.3).
        let expected = 0.3_f64.ln();
        let tol = 1e-7;
        assert!(
            (logp - expected).abs() < tol,
            "For index 1, expected ln(0.3) ~ {}, got {}",
            expected,
            logp
        );
    }

    #[test]
    fn test_target_for_categorical_out_of_range() {
        let probs = vec![0.2_f64, 0.3, 0.5];
        let cat = Categorical::new(probs);
        // Calling unnorm_log_prob with an index that's out of bounds (e.g. 3)
        // should return negative infinity.
        let logp = cat.unnorm_log_prob(&[3]);
        assert_eq!(
            logp,
            f64::NEG_INFINITY,
            "Expected negative infinity for out-of-range index, got {}",
            logp
        );
    }

    #[test]
    fn test_gaussian2d_log_prob() {
        let mean = arr1(&[0.0, 0.0]);
        let cov = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let gauss = Gaussian2D { mean, cov };

        let theta = vec![0.5, -0.5];
        let computed_logp = gauss.log_prob(&theta);

        let expected_logp = -2.0878770664093453;

        let tol = 1e-10;
        assert!(
            (computed_logp - expected_logp).abs() < tol,
            "Computed log probability ({}) differs from expected ({}) by more than tolerance ({})",
            computed_logp,
            expected_logp,
            tol
        );
    }
}
