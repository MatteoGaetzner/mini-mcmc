# Mini MCMC Library

A small (and growing) Rust library for **Markov Chain Monte Carlo (MCMC)** methods.

## Overview

This library provides:

- **Metropolis-Hastings**: A generic implementation suitable for various target distributions and proposal mechanisms.
- **Distributions**: Handy Gaussian and isotropic Gaussian implementations, along with traits for defining custom log-prob functions.

## Roadmap

- **Parallel Chains**: Run multiple Metropolis-Hastings Markov chains in parallel. ✅
- **Gibbs Sampling**: A component-wise MCMC approach for higher-dimensional problems.
- **Hamiltonian Monte Carlo (HMC)**: A gradient-based method for efficient exploration.
- **No-U-Turn Sampler (NUTS)**: An extension of HMC that removes the need to choose path lengths.

## Structure

- **`src/lib.rs`**: The main library entry point—exports MCMC functionality.
- **`src/distributions.rs`**: Target distributions (e.g., multivariate Gaussians) and proposal distributions.
- **`src/metrophast.rs`**: The Metropolis-Hastings algorithm implementation.
- **`src/bin/demo.rs`**: Example usage demonstrating 2D Gaussian sampling and plotting.

## Usage

1.  **Build** (Library + Demo):

    ```sh
    cargo build --release
    ```

2.  **Run the Demo**:

    ```sh
    cargo run --release --bin demo
    ```

    Prints basic statistics of the MCMC chain (e.g., estimated mean).
    Saves a scatter plot of sampled points in samples.png.

## License

Apache License, Version 2.0

This project includes code from the kolmogorov_smirnov project, which is licensed under the Apache License, Version 2.0.
