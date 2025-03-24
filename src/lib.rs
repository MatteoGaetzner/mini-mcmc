//! # Mini MCMC
//!
//! A compact Rust library offering **Markov Chain Monte Carlo (MCMC)** methods,
//! including **Hamiltonian Monte Carlo (HMC)**, **Metropolis–Hastings**, and
//! **Gibbs Sampling** for both discrete and continuous targets.
//!
//! ## Example 1: Sampling a 2D Gaussian (Metropolis–Hastings)
//!
//! ```rust
//! use mini_mcmc::core::ChainRunner;
//! use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
//! use mini_mcmc::metropolis_hastings::MetropolisHastings;
//! use ndarray::{arr1, arr2};
//!
//! let target = Gaussian2D {
//!     mean: arr1(&[0.0, 0.0]),
//!     cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
//! };
//! let proposal = IsotropicGaussian::new(1.0);
//! let initial_state = [0.0, 0.0];
//!
//! let mut mh = MetropolisHastings::new(target, proposal, &initial_state, 4);
//! let samples = mh.run(1000, 100).unwrap();
//! println!("Metropolis–Hastings samples shape: {:?}", samples.shape());
//! ```
//!
//! ## Example 2: Sampling a 3D Rosenbrock (HMC)
//!
//! ```rust
//! use burn::tensor::Element;
//! use burn::{backend::Autodiff, prelude::Tensor};
//! use mini_mcmc::hmc::{GradientTarget, HMC};
//! use num_traits::Float;
//!
//! /// The 3D Rosenbrock distribution.
//! ///
//! /// For a point x = (x₁, x₂, x₃), the log probability is defined as the negative of
//! /// the sum of two Rosenbrock terms:
//! ///
//! ///   f(x) = 100*(x₂ - x₁²)² + (1 - x₁)² + 100*(x₃ - x₂²)² + (1 - x₂)²
//! ///
//! /// This implementation generalizes to d dimensions, but here we use it for 3D.
//! struct RosenbrockND {}
//!
//! impl<T, B> GradientTarget<T, B> for RosenbrockND
//! where
//!     T: Float + std::fmt::Debug + Element,
//!     B: burn::tensor::backend::AutodiffBackend,
//! {
//!     fn log_prob_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1> {
//!         // Assume positions has shape [n_chains, d] with d = 3.
//!         let k = positions.dims()[0] as i64;
//!         let n = positions.dims()[1] as i64;
//!         let low = positions.clone().slice([(0, k), (0, n - 1)]);
//!         let high = positions.clone().slice([(0, k), (1, n)]);
//!         let term_1 = (high - low.clone().powi_scalar(2))
//!             .powi_scalar(2)
//!             .mul_scalar(100);
//!         let term_2 = low.neg().add_scalar(1).powi_scalar(2);
//!         -(term_1 + term_2).sum_dim(1).squeeze(1)
//!     }
//! }
//!
//! // Use the CPU backend wrapped in Autodiff (e.g., NdArray).
//! type BackendType = Autodiff<burn::backend::NdArray>;
//!
//! // Create the 3D Rosenbrock target.
//! let target = RosenbrockND {};
//!
//! // Define initial positions for 6 chains (each a 3D point).
//! let initial_positions = vec![vec![1.0_f32, 2.0_f32, 3.0_f32]; 6];
//!
//! // Create the HMC sampler with a step size of 0.01 and 50 leapfrog steps.
//! let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
//!     target,
//!     initial_positions,
//!     0.032,
//!     50,
//! );
//!
//! // Run the sampler for 1100 iterations, discard 100
//! let samples = sampler.run(1000, 100);
//!
//! // Print the shape of the collected samples.
//! println!("Collected samples with shape: {:?}", samples.dims());
//! ```
//!
//! ## Example 3: Sampling a Poisson Distribution (Discrete)
//!
//! ```rust
//! use mini_mcmc::core::ChainRunner;
//! use mini_mcmc::distributions::{Proposal, Target};
//! use mini_mcmc::metropolis_hastings::MetropolisHastings;
//! use rand::Rng;
//!
//! #[derive(Clone)]
//! struct PoissonTarget {
//!     lambda: f64,
//! }
//!
//! impl Target<usize, f64> for PoissonTarget {
//!     fn unnorm_log_prob(&self, theta: &[usize]) -> f64 {
//!         let k = theta[0];
//!         -self.lambda + (k as f64) * self.lambda.ln() - ln_factorial(k as u64)
//!     }
//! }
//!
//! #[derive(Clone)]
//! struct NonnegativeProposal;
//!
//! impl Proposal<usize, f64> for NonnegativeProposal {
//!     fn sample(&mut self, current: &[usize]) -> Vec<usize> {
//!         let x = current[0];
//!         if x == 0 {
//!             vec![1]
//!         } else {
//!             let step_up = rand::thread_rng().gen_bool(0.5);
//!             vec![if step_up { x + 1 } else { x - 1 }]
//!         }
//!     }
//!
//!     fn log_prob(&self, from: &[usize], to: &[usize]) -> f64 {
//!         let (x, y) = (from[0], to[0]);
//!         if x == 0 && y == 1 {
//!             0.0
//!         } else if x > 0 && (y == x + 1 || y == x - 1) {
//!             (0.5_f64).ln()
//!         } else {
//!             f64::NEG_INFINITY
//!         }
//!     }
//!     fn set_seed(self, _seed: u64) -> Self {
//!         self
//!     }
//! }
//!
//! fn ln_factorial(k: u64) -> f64 {
//!     (1..=k).map(|v| (v as f64).ln()).sum()
//! }
//!
//! let target = PoissonTarget { lambda: 4.0 };
//! let proposal = NonnegativeProposal;
//! let initial_state = [0];
//!
//! let mut mh = MetropolisHastings::new(target, proposal, &initial_state, 1);
//! let samples = mh.run(5000, 100).unwrap();
//! println!("Poisson samples shape: {:?}", samples.shape());
//! ```
//!
//! For more complete implementations (including Gibbs sampling and I/O helpers),
//! see the `examples/` directory.
//!
//! ## Features
//! - **Parallel Chains** for improved throughput
//! - **Progress Indicators** (acceptance rates, iteration counts)
//! - **Common Distributions** (e.g. Gaussian) plus easy traits for custom log‐prob
//! - **Optional I/O** (CSV, Arrow, Parquet) and GPU sampling (WGPU)
//!
//! ## Roadmap
//! - No-U-Turn Sampler (NUTS)
//! - Rank-Normalized R-hat diagnostics
//! - Ensemble Slice Sampling (ESS)
//! - Effective Sample Size estimation

pub mod core;
mod dev_tools;
pub mod distributions;
pub mod gibbs;
pub mod hmc;
pub mod io;
pub mod ks_test;
pub mod metropolis_hastings;
pub mod stats;
