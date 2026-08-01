//! # Mini MCMC
//!
//! A compact Rust library offering **Markov Chain Monte Carlo (MCMC)** methods, including
//! **No-U-Turn Sampler (NUTS)**, **Hamiltonian Monte Carlo (HMC)**, **Metropolis–Hastings**, and
//! **Gibbs Sampling** for both discrete and continuous targets.
//!
//! ## Getting Started
//!
//! To use this library, add it to your project:
//! ```bash
//! cargo add mini-mcmc
//! ```
//!
//! The library provides three main sampling approaches:
//! 1. **No-U-Turn Sampler (NUTS)**: For continuous distributions with gradients. You need to provide:
//!    - A target distribution implementing the `GradientTarget` trait (via `burn` autodiff), or
//!      [`distributions::ManualGradientTarget`] (analytic gradient, no `burn` tensors) for use with
//!      [`nuts::ManualNUTS`] — see Example 5 below.
//! 2. **Hamiltonian Monte Carlo (HMC)**: For continuous distributions with gradients. You need to provide:
//!    - A target distribution implementing the `BatchedGradientTarget` trait (via `burn` autodiff), or
//!      [`distributions::ManualGradientTarget`] for use with [`hmc::ManualHMC`].
//! 3. **Metropolis-Hastings**: For general-purpose sampling. You need to provide:
//!    - A target distribution implementing the `Target` trait
//!    - A proposal distribution implementing the `Proposal` trait
//! 4. **Gibbs Sampling**: For sampling when conditional distributions are available. You need to provide:
//!    - A distribution implementing the `Conditional` trait
//!
//! ## Example 1: Sampling a 2D Rosenbrock (NUTS)
//!
//! ```rust
//! use burn::backend::Autodiff;
//! use burn::prelude::Tensor;
//! use mini_mcmc::core::init;
//! use mini_mcmc::distributions::Rosenbrock2D;
//! use mini_mcmc::nuts::NUTS;
//!
//! // CPU backend with autodiff (NdArray).
//! type BackendType = Autodiff<burn::backend::NdArray>;
//!
//! // 2D Rosenbrock target (a=1, b=100).
//! let target = Rosenbrock2D { a: 1.0_f32, b: 100.0_f32 };
//!
//! // 4 independent chains starting at (1.0,2.0).
//! let initial_positions = init::<f32>(4, 2);
//!
//! // NUTS with 0.95 target‐accept and a fixed seed.
//! let mut sampler = NUTS::new(target, initial_positions, 0.95)
//!     .set_seed(42);
//!
//! // 400 burn-in + 400 samples.
//! let n_collect = 400;
//! let n_discard = 400;
//!
//! let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);
//!
//! println!(
//!     "Collected {} chains × {} samples × 2 dims",
//!     sample.dims()[0],
//!     sample.dims()[1],
//! );
//! ```
//!
//! ## Example 2: Sampling a 3D Rosenbrock (HMC)
//!
//! ```rust
//! use burn::tensor::Element;
//! use burn::{backend::Autodiff, prelude::Tensor};
//! use mini_mcmc::hmc::HMC;
//! use mini_mcmc::distributions::BatchedGradientTarget;
//! use mini_mcmc::core::init;
//! use num_traits::Float;
//!
//! /// The 3D Rosenbrock distribution.
//! ///
//! /// For a point x = (x₁, x₂, x₃), the log density is defined as
//! ///
//! ///   f(x) = 100*(x₂ - x₁²)² + (1 - x₁)² + 100*(x₃ - x₂²)² + (1 - x₂)².
//! ///
//! /// This implementation generalizes to d dimensions, but here we use it for 3D.
//! struct RosenbrockND {}
//!
//! impl<T, B> BatchedGradientTarget<T, B> for RosenbrockND
//! where
//!     T: Float + std::fmt::Debug + Element,
//!     B: burn::tensor::backend::AutodiffBackend,
//! {
//!     fn unnorm_logp_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1> {
//!         // Assume positions has shape [n_chains, d] with d = 3.
//!         let k = positions.dims()[0];
//!         let n = positions.dims()[1];
//!         let low = positions.clone().slice([0..k, 0..n-1]);
//!         let high = positions.clone().slice([0..k, 1..n]);
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
//! let initial_positions = init(6, 3);
//!
//! // Create the HMC sampler with a step size of 0.01 and 5 leapfrog steps.
//! let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
//!     target,
//!     initial_positions,
//!     0.032,
//!     5,
//! );
//!
//! // Run the sampler for 123+45 iterations, discard 45 burnin observations
//! let sample = sampler.run(123, 45);
//!
//! // Print the shape of the collected sample.
//! println!("Collected sample with shape: {:?}", sample.dims());
//! ```
//!
//!
//! See [`examples/minimal_nuts.rs`](examples/minimal_nuts.rs) for the full version with diagnostics.
//!
//! ## Example 3: Sampling a 2D Gaussian (Metropolis–Hastings, Continuous)
//!
//! ```rust
//! use mini_mcmc::core::{ChainRunner, init};
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
//! let mut mh = MetropolisHastings::new(target, proposal, init(4, 2));
//! let sample = mh.run(1000, 100).unwrap();
//! println!("Metropolis–Hastings sample shape: {:?}", sample.shape());
//! ```
//!
//! ## Example 4: Sampling a Poisson Distribution (Metropolis-Hastings, Discrete)
//!
//! ```rust
//! use mini_mcmc::core::{ChainRunner, init};
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
//!     /// unnorm_logp(k) = log( p(k) ), ignoring normalizing constants if you wish.
//!     /// For Poisson(k|lambda) = exp(-lambda) * (lambda^k / k!)
//!     /// so log p(k) = -lambda + k*ln(lambda) - ln(k!)
//!     /// which is enough to do MH acceptance.
//!     fn unnorm_logp(&self, theta: &[usize]) -> f64 {
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
//!             let step_up = rand::rng().gen_bool(0.5);
//!             vec![if step_up { x + 1 } else { x - 1 }]
//!         }
//!     }
//!
//!     fn logp(&self, from: &[usize], to: &[usize]) -> f64 {
//!         let (x, y) = (from[0], to[0]);
//!         if x == 0 && y == 1 {
//!             0.0
//!         } else if x > 0 && (y == x + 1 || y == x - 1) {
//!             (0.5_f64).ln()
//!         } else {
//!             f64::NEG_INFINITY
//!         }
//!     }
//!
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
//! let initial_state = vec![vec![0]];
//!
//! let mut mh = MetropolisHastings::new(target, proposal, initial_state);
//! let sample = mh.run(5000, 100).unwrap();
//! println!("Poisson sample shape: {:?}", sample.shape());
//! ```
//!
//! ## Example 5: Sampling a 2D Rosenbrock with an analytic gradient (burn-free NUTS)
//!
//! NUTS and HMC don't require `burn`: if you already have a closed-form gradient,
//! implement [`distributions::ManualGradientTarget`] and use [`nuts::ManualNUTS`] (or
//! [`hmc::ManualHMC`]) instead. No tensors, no autodiff, no per-step allocation —
//! `unnorm_logp_and_grad_into` writes the gradient into a buffer the sampler reuses
//! across leapfrog steps.
//!
//! ```rust
//! use mini_mcmc::core::{ChainRunner, init};
//! use mini_mcmc::distributions::ManualGradientTarget;
//! use mini_mcmc::nuts::ManualNUTS;
//!
//! #[derive(Clone)]
//! struct Rosenbrock2D { a: f64, b: f64 }
//!
//! impl ManualGradientTarget<f64> for Rosenbrock2D {
//!     fn unnorm_logp_and_grad_into(&self, position: &[f64], grad: &mut [f64]) -> f64 {
//!         let (x, y) = (position[0], position[1]);
//!         let a_minus_x = self.a - x;
//!         let y_minus_x2 = y - x * x;
//!         grad[0] = 2.0 * a_minus_x + 4.0 * self.b * x * y_minus_x2;
//!         grad[1] = -2.0 * self.b * y_minus_x2;
//!         -(a_minus_x * a_minus_x + self.b * y_minus_x2 * y_minus_x2)
//!     }
//! }
//!
//! let target = Rosenbrock2D { a: 1.0, b: 100.0 };
//! let n_discard = 20; // also the step-size adaptation window, see `ManualNUTS::new`
//! let mut sampler = ManualNUTS::new(target, init(4, 2), 0.8, n_discard).set_seed(42);
//! let sample = sampler.run(100, n_discard).unwrap();
//! println!("Collected sample shape: {:?}", sample.shape());
//! ```
//!
//! For more complete implementations (including Gibbs sampling and I/O helpers),
//! see the `examples/` directory.
//!
//! ## Features
//! - **Parallel Chains** for improved throughput
//! - **Progress Indicators** (acceptance rates, max Rhat, iteration counts)
//! - **Common Distributions** (e.g. Gaussian) plus easy traits for custom log‐prob
//! - **Optional I/O** (CSV, Arrow, Parquet) and GPU support (WGPU)
//! - **Effective Sample Size (ESS)** estimation following STAN's methodology
//! - **R-hat Diagnostics** for convergence monitoring
//!
//! ## Roadmap
//! - Rank-Normalized R-hat diagnostics
//! - Ensemble Slice Sampling (ESS)

// Lets `#[coverage(off)]` (below, on slow `#[ignore]`d correctness tests) compile on
// stable as a no-op, while actually excluding those functions from coverage accounting
// when `cargo llvm-cov` runs on nightly. See taiki-e/cargo-llvm-cov's README.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

pub mod core;
mod dev_tools;
pub mod distributions;
pub mod gibbs;
pub mod hmc;
pub mod io;
mod leapfrog;
pub mod metropolis_hastings;
pub mod nuts;
pub mod stats;
