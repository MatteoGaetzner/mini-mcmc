# Mini MCMC

![tests](https://github.com/MatteoGaetzner/mini-mcmc/actions/workflows/general.yml/badge.svg)
![security](https://github.com/MatteoGaetzner/mini-mcmc/actions/workflows/audit.yml/badge.svg)
[![codecov](https://codecov.io/gh/MatteoGaetzner/mini-mcmc/graph/badge.svg?token=IDLWGMMUFI)](https://codecov.io/gh/MatteoGaetzner/mini-mcmc)

A compact Rust library for **Markov Chain Monte Carlo (MCMC)** methods with GPU support.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
mini-mcmc = "0.4.2"
```

Then `use mini_mcmc` in your Rust code.

## Example: Sampling From a 2D Gaussian

```rust
use mini_mcmc::core::ChainRunner;
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
use mini_mcmc::metropolis_hastings::MetropolisHastings;
use ndarray::{arr1, arr2};

fn main() {
    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let initial_state = [0.0, 0.0];

    // Create a MH sampler with 4 parallel chains
    let mut mh = MetropolisHastings::new(target, proposal, &initial_state, 4);

    // Run the sampler for 1,100 steps, discarding the first 100 as burn-in
    let samples = mh.run(1000, 100).unwrap();

    // We should have 1000 * 4 = 3600 samples
    assert_eq!(samples.shape()[0], 4);
    assert_eq!(samples.shape()[1], 1000);
}
```

You can also find this example at `examples/minimal_mh.rs`.

## Example: Sampling From a Custom Distribution

Below we define a custom Poisson distribution for nonnegative integer states $\{0,1,2,\dots\}$ and a basic random-walk proposal. We then run Metropolis–Hastings to sample from this distribution, collecting frequencies of $k$ after some burn-in:

```rust
use mini_mcmc::core::ChainRunner;
use mini_mcmc::distributions::{Proposal, Target};
use mini_mcmc::metropolis_hastings::MetropolisHastings;
use rand::Rng; // for thread_rng

/// A Poisson(\lambda) distribution, seen as a discrete target over k=0,1,2,...
#[derive(Clone)]
struct PoissonTarget {
    lambda: f64,
}

impl Target<usize, f64> for PoissonTarget {
    /// unnorm_log_prob(k) = log( p(k) ), ignoring normalizing constants if you wish.
    /// For Poisson(k|lambda) = exp(-lambda) * (lambda^k / k!)
    /// so log p(k) = -lambda + k*ln(lambda) - ln(k!)
    /// which is enough to do MH acceptance.
    fn unnorm_log_prob(&self, theta: &[usize]) -> f64 {
        let k = theta[0];
        let kf = k as f64;
        // If you like, you can omit -ln(k!) if you only need "unnormalized"—but including
        // it can improve acceptance ratio numerically. Here we keep the full log pmf.
        -self.lambda + kf * self.lambda.ln() - ln_factorial(k as u64)
    }
}

/// A simple random-walk proposal in the nonnegative integers:
/// - If current_state=0, propose 0 -> 1 always
/// - Otherwise propose x->x+1 or x->x-1 with p=0.5 each
#[derive(Clone)]
struct NonnegativeProposal;

impl Proposal<usize, f64> for NonnegativeProposal {
    fn sample(&mut self, current: &[usize]) -> Vec<usize> {
        let x = current[0];
        if x == 0 {
            // can't go negative; always move to 1
            vec![1]
        } else {
            // 50% chance to do x+1, 50% x-1
            let flip = rand::thread_rng().gen_bool(0.5);
            let next = if flip { x + 1 } else { x - 1 };
            vec![next]
        }
    }

    /// log_prob(x->y):
    ///  - if x=0 and y=1, p=1 => log p=0
    ///  - if x>0, then y in {x+1, x-1} => p=0.5 => log(0.5)
    ///  - otherwise => -∞ (impossible transition)
    fn log_prob(&self, from: &[usize], to: &[usize]) -> f64 {
        let x = from[0];
        let y = to[0];
        if x == 0 {
            if y == 1 {
                0.0 // ln(1.0)
            } else {
                f64::NEG_INFINITY
            }
        } else {
            // x>0
            if y == x + 1 || y + 1 == x {
                // y in {x+1, x-1} => prob=0.5 => ln(0.5)
                (0.5_f64).ln()
            } else {
                f64::NEG_INFINITY
            }
        }
    }

    fn set_seed(self, _seed: u64) -> Self {
        // no custom seeding logic here
        self
    }
}

// A small helper for computing ln(k!)
fn ln_factorial(k: u64) -> f64 {
    if k < 2 {
        0.0
    } else {
        let mut acc = 0.0;
        for i in 1..=k {
            acc += (i as f64).ln();
        }
        acc
    }
}

fn main() {
    // We'll do Poisson with lambda=4.0, for instance
    let target = PoissonTarget { lambda: 4.0 };

    // We'll have a random-walk in nonnegative integers
    let proposal = NonnegativeProposal;

    // Start the chain at k=0
    let initial_state = [0usize];

    // Create Metropolis–Hastings with 1 chain (or more, up to you)
    let mut mh = MetropolisHastings::new(target, proposal, &initial_state, 1);

    // Collect 10,000 samples and use 1,000 for burn-in (not returned).
    let samples = mh
        .run(10_000, 1_000)
        .expect("Expected generating samples to succeed");
    let chain0 = samples.to_shape(10_000).unwrap();
    println!("Elements in chain: {}", chain0.len());

    // Tally frequencies of each k up to some cutoff
    let cutoff = 20; // enough to see the mass near lambda=4
    let mut counts = vec![0usize; cutoff + 1];
    for row in chain0.iter() {
        let k = *row;
        if k <= cutoff {
            counts[k] += 1;
        }
    }

    let total = chain0.len();
    println!("Frequencies for k=0..{cutoff}, from chain after burn-in:");
    for (k, &cnt) in counts.iter().enumerate() {
        let freq = cnt as f64 / total as f64;
        println!("k={k:2}: freq ~ {freq:.3}");
    }

    // We might compare these frequencies to the theoretical Poisson(4.0) pmf
    // in a quick check.
    println!("Done sampling Poisson(4).");
}
```

You can also find this example at `examples/poisson_mh.rs`.

### Explanation

- **`PoissonTarget`** implements `Target<usize, f64>` for a discrete Poisson($\lambda$) distribution:\
  $$p(k) = e^{-\lambda} \frac{\lambda^k}{k!},\quad k=0,1,2,\ldots$$\
  The log form of it is $\log p(k) = -\lambda + k \log \lambda - \log k!$.

- **`NonnegativeProposal`** provides a random-walk in the set $\{0,1,2,\dots\}$:

  - If $x=0$, propose $1$ with probability $1$.
  - If $x>0$, propose $x+1$ or $x-14$ with probability $0.5$ each.
  - `log_prob` returns $\ln(0.5)$ for the possible moves, or $-\infty$ for
    impossible moves.

- **Usage**:  
  We start the chain at $k=0$, run 11,000 iterations discarding 1,000 as burn-in, and tally the final sample frequencies for $k=0 \dots 20$. They should approximate the Poisson(4.0) distribution (peak around $k=4$).

With this example, you can see how to use **mini_mcmc** for **unbounded** discrete distributions via a custom random-walk proposal and a log‐PMF.

Below is an additional documentation section that you can add to your README. It first gives a minimal version of the `rosenbrock3d_hmc.rs` example for sampling using HMC. (Note that the full example also plots the sampled data interactively using Plotly.)

---

## Example: Sampling from a 3D Rosenbrock Distribution Using HMC

The following minimal example demonstrates how to create and run an HMC sampler to sample from a 3D Rosenbrock distribution. In this example, we construct an HMC sampler, run it for a fixed number of iterations, and print the shape of the collected samples. The corresponding file can also be found at [`examples/minimal_hmc.rs`](examples/minimal_hmc.rs). For a complete example—including interactive 3D plotting with Plotly, refer to [`examples/rosenbrock3d_hmc.rs`](examples/rosenbrock3d_hmc.rs).

```rust
use burn::tensor::Element;
use burn::{backend::Autodiff, prelude::Tensor};
use mini_mcmc::hmc::{GradientTarget, HMC};
use num_traits::Float;

/// The 3D Rosenbrock distribution.
///
/// For a point x = (x₁, x₂, x₃), the log probability is defined as the negative of
/// the sum of two Rosenbrock terms:
///
///   f(x) = 100*(x₂ - x₁²)² + (1 - x₁)² + 100*(x₃ - x₂²)² + (1 - x₂)²
///
/// This implementation generalizes to d dimensions, but here we use it for 3D.
struct RosenbrockND {}

impl<T, B> GradientTarget<T, B> for RosenbrockND
where
    T: Float + std::fmt::Debug + Element,
    B: burn::tensor::backend::AutodiffBackend,
{
    fn log_prob_batch(&self, positions: &Tensor<B, 2>) -> Tensor<B, 1> {
        // Assume positions has shape [n_chains, d] with d = 3.
        let k = positions.dims()[0] as i64;
        let n = positions.dims()[1] as i64;
        let low = positions.clone().slice([(0, k), (0, n - 1)]);
        let high = positions.clone().slice([(0, k), (1, n)]);
        let term_1 = (high - low.clone().powi_scalar(2))
            .powi_scalar(2)
            .mul_scalar(100);
        let term_2 = low.neg().add_scalar(1).powi_scalar(2);
        -(term_1 + term_2).sum_dim(1).squeeze(1)
    }
}

fn main() {
    // Use the CPU backend wrapped in Autodiff (e.g., NdArray).
    type BackendType = Autodiff<burn::backend::NdArray>;

    // Create the 3D Rosenbrock target.
    let target = RosenbrockND {};

    // Define initial positions for 6 chains (each a 3D point).
    let initial_positions = vec![vec![1.0_f32, 2.0_f32, 3.0_f32]; 6];

    // Create the HMC sampler with a step size of 0.01 and 50 leapfrog steps.
    let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
        target,
        initial_positions,
        0.032,
        50,
    );

    // Run the sampler for 1100 iterations, discard 100
    let samples = sampler.run(1000, 100);

    // Print the shape of the collected samples.
    println!("Collected samples with shape: {:?}", samples.dims());
}
```

## Overview

This library provides implementations of

- **Hamiltonian Monte Carlo (HMC)**: an MCMC method that efficiently samples by simulating Hamiltonian dynamics using gradients of the target distribution.
- **Metropolis-Hastings**: an MCMC algorithm that samples from a distribution by proposing candidates and probabilistically accepting or rejecting them.
- **Gibbs Sampling**: an MCMC method that iteratively samples each variable from its conditional distribution given all other variables.

with

- **Implementations of Common Distributions**: featuring handy Gaussian and isotropic Gaussian implementations, along with traits for defining custom log-prob functions.
- **Parallelization**: for running multiple Markov chains in parallel.
- **Progress Bars**: that show progress of MCMC algorithms with convergence
  statistics and acceptance rates.
- **Support for Discrete & Continuous Distributions**: for example, Metropolis-Hastings- and Gibbs Samplers can sample from continuous and discrete target distributions.
- **Generic Datatypes**: enable sampling of vectors with various integer or floating point types.

## Roadmap

- **No-U-Turn Sampler (NUTS)**: An extension of HMC that removes the need to choose path lengths.
- **Rank Normalized Rhat**: Modern convergence diagnostic, see [paper](https://arxiv.org/abs/1903.08008).
- **Ensemble Slice Sampling (ESS)**: Efficient gradient-free sampler, see [paper](https://arxiv.org/abs/2002.06212).
- **Effective Size Estimation**: Online estimation of effective sample size for
  early stopping.

## Structure

- **`src/lib.rs`**: The main library entry point—exports MCMC functionality.
- **`src/distributions.rs`**: Target distributions (e.g., multivariate Gaussians) and proposal distributions.
- **`src/metropolis_hastings.rs`**: The Metropolis-Hastings algorithm implementation.
- **`src/gibbs.rs`**: The Gibbs sampling algorithm implementation.
- **`examples/`**: Examples on how to use this library.
- **`src/io/arrow.rs`**: Helper functions for saving samples as Apache Arrow files. Enable via `arrow` feature.
- **`src/io/parquet.rs`**: Helper functions for saving samples as Apache Parquet files. Enable via `parquet` feature.
- **`src/io/csv.rs`**: Helper functions for saving samples as Apache Parquet files. Enable via `csv` feature.

## Usage (Local)

1. **Build** (Library + Demo):

   ```sh
   cargo build --release
   ```

2. **Run the Demo**:
   ```sh
   cargo run --release --example gauss_mh --features parquet
   ```
   Prints basic statistics of the MCMC chain (e.g., estimated mean).
   Saves a scatter plot of sampled points in `scatter_plot.png` and a Parquet file `samples.parquet`.

## Optional Features

- `csv`: Enables CSV I/O for samples.
- `arrow` / `parquet`: Enables Apache Arrow / Parquet I/O.
- `wgpu`: Enables GPU accelerated sampling for gradient based samplers using burn's WGPU backend.
  In the HMC example above, you only have to replace line
  ```rust
  type BackendType = Autodiff<burn::backend::NdArray>;
  ```
  with
  ```rust
  type BackendType = Autodiff<burn::backend::Wgpu>;
  ```
  Depending on the number of parallel chains, dimensionality of your
  sample space and complexity of evaluating the unnormalized log density of
  your target distribution it might be more efficient to stick with the CPU (NdArray) backend.
- By default, all features are **disabled**.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.  
This project includes code from the `kolmogorov_smirnov` project, licensed under Apache 2.0 as noted in [NOTICE](NOTICE).
