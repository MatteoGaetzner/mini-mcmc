# Mini MCMC Library

[![Build & Test](https://github.com/MatteoGaetzner/mini_mcmc/workflows/Rust/badge.svg?branch=main)](https://github.com/MatteoGaetzner/mini_mcmc/actions?query=workflow%3ARust+branch%3Amain)
[![Coverage Status](https://codecov.io/gh/MatteoGaetzner/mini_mcmc/branch/main/graph/badge.svg)](https://codecov.io/gh/MatteoGaetzner/mini_mcmc)

A small (and growing) Rust library for **Markov Chain Monte Carlo (MCMC)** methods.

## Installation

Once published on crates.io, add the following to your `Cargo.toml`:

```toml
[dependencies]
mini-mcmc = "0.1.0"
```

Then you can `use mini_mcmc` in your Rust code.

## Quick Example

```rust
use mini_mcmc::metrohast::MetropolisHastings;
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};

fn main() {
    let target = Gaussian2D {
        mean: [0.0, 0.0].into(),
        cov: [[1.0, 0.0], [0.0, 1.0]].into(),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let initial_state = vec![0.0, 0.0];

    // Create a MH sampler
    let mut mh = MetropolisHastings::new(target, proposal, initial_state, 1);

    // Run the sampler for 1,000 steps, discarding the first 100 as burn-in
    let samples = mh.run(1000, 100);

    println!("Number of post-burn-in samples: {}", samples[0].len());
}
```

## Overview

This library provides:

- **Metropolis-Hastings**: A generic implementation suitable for various target distributions and proposal mechanisms.
- **Distributions**: Handy Gaussian and isotropic Gaussian implementations, along with traits for defining custom log-prob functions.

## Roadmap

- **Parallel Chains**: Run multiple Metropolis-Hastings Markov chains in parallel. ✅
- **Discrete & Continuous Distributions**: Get Metropolis-Hastings to work for continuous and discrete distributions. ✅
- **Generic Datatypes**: Support sampling vectors of arbitrary integer or floating point types. ✅
- **Gibbs Sampling**: A component-wise MCMC approach for higher-dimensional problems.
- **Hamiltonian Monte Carlo (HMC)**: A gradient-based method for efficient exploration.
- **No-U-Turn Sampler (NUTS)**: An extension of HMC that removes the need to choose path lengths.

## Structure

- **`src/lib.rs`**: The main library entry point—exports MCMC functionality.
- **`src/distributions.rs`**: Target distributions (e.g., multivariate Gaussians) and proposal distributions.
- **`src/metrohast.rs`**: The Metropolis-Hastings algorithm implementation.
- **`examples/demo.rs`**: Example usage demonstrating 2D Gaussian sampling and plotting.

## Usage (Local)

1. **Build** (Library + Demo):

   ```sh
   cargo build --release
   ```

2. **Run the Demo**:
   ```sh
   cargo run --release --bin demo
   ```
   Prints basic statistics of the MCMC chain (e.g., estimated mean).
   Saves a scatter plot of sampled points in `scatter_plot.png` and a Parquet file `samples.parquet`.

## Optional Features

- `csv`: Enables CSV I/O for samples.
- `arrow` / `parquet`: Enables Apache Arrow / Parquet I/O.
- By default, all features are enabled. You can disable them if you want a smaller build.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.  
This project includes code from the `kolmogorov_smirnov` project, licensed under Apache 2.0 as noted in [NOTICE](NOTICE).
