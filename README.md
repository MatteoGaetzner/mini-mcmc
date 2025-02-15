# Mini MCMC

![tests](https://github.com/MatteoGaetzner/mini-mcmc/actions/workflows/general.yml/badge.svg)
![security](https://github.com/MatteoGaetzner/mini-mcmc/actions/workflows/audit.yml/badge.svg)
[![codecov](https://codecov.io/gh/MatteoGaetzner/mini-mcmc/graph/badge.svg?token=IDLWGMMUFI)](https://codecov.io/gh/MatteoGaetzner/mini-mcmc)

A small (and growing) Rust library for **Markov Chain Monte Carlo (MCMC)** methods.

## Installation

Once published on crates.io, add the following to your `Cargo.toml`:

```toml
[dependencies]
mini-mcmc = "0.2.0"
```

Then you can `use mini_mcmc` in your Rust code.

## Example: Sampling From a 2D Gaussian

```rust
use mini_mcmc::core::ChainRunner;
use mini_mcmc::metropolis_hastings::MetropolisHastings;
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};

fn main() {
        let target = Gaussian2D {
            mean: [0.0, 0.0].into(),
            cov: [[1.0, 0.0], [0.0, 1.0]].into(),
        };
        let proposal = IsotropicGaussian::new(1.0);
        let initial_state = [0.0, 0.0];

        // Create a MH sampler with 4 parallel chains
        let mut mh = MetropolisHastings::new(target, proposal, &initial_state, 4);

        // Run the sampler for 1,000 steps, discarding the first 100 as burn-in
        let samples = mh.run(1000, 100);

        // We should have 900 * 4 = 3600 samples
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].nrows(), 900); // samples[0] is a nalgebra::DMatrix
}
```

You can also find this example at `examples/minimal_mh.rs`.

## Example: Sampling From a Custom Distribution

Below we define a custom Poisson distribution for nonnegative integer states $\{0,1,2,\dots\}$ and a simple random-walk proposal. We then run Metropolis–Hastings to sample from this distribution, collecting frequencies of $k$ after some burn-in:

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

    // Run 10_000 steps, discarding first 1_000
    let samples = mh.run(10_000, 1_000);
    let chain0 = &samples[0];
    println!("Chain shape = {} x {}", chain0.nrows(), chain0.ncols());

    // Tally frequencies of each k up to some cutoff
    let cutoff = 20; // enough to see the mass near lambda=4
    let mut counts = vec![0usize; cutoff + 1];
    for row in chain0.row_iter() {
        let k = row[0];
        if k <= cutoff {
            counts[k] += 1;
        }
    }

    let total = chain0.nrows();
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
  - `log_prob` returns $\ln(0.5)$ for the valid moves, or $-\infty$ for invalid moves.

- **Usage**:  
  We start the chain at $k=0$, run 10,000 iterations discarding 1,000 as burn-in, and tally the final sample frequencies for $k=0 \dots 20$. They should approximate the Poisson(4.0) distribution (peak around $k=4$).

With this example, you can see how to use **mini_mcmc** for **unbounded** discrete distributions via a custom random-walk proposal and a log‐PMF.

## Overview

This library provides:

- **Metropolis-Hastings**: A generic implementation suitable for various target distributions and proposal mechanisms.
- **Distributions**: Handy Gaussian and isotropic Gaussian implementations, along with traits for defining custom log-prob functions.

## Roadmap

- **Parallel Chains**: Run multiple Metropolis-Hastings Markov chains in parallel. ✅
- **Discrete & Continuous Distributions**: Get Metropolis-Hastings to work for continuous and discrete distributions. ✅
- **Generic Datatypes**: Support sampling vectors of arbitrary integer or floating point types. ✅
- **Gibbs Sampling**: A component-wise MCMC approach for higher-dimensional problems. ✅
- **Hamiltonian Monte Carlo (HMC)**: A gradient-based method for efficient exploration.
- **No-U-Turn Sampler (NUTS)**: An extension of HMC that removes the need to choose path lengths.
- **Ensemble Slice Sampling (ESS)**: Efficient gradient-free sampler, see [paper](https://arxiv.org/abs/2002.06212).

## Structure

- **`src/lib.rs`**: The main library entry point—exports MCMC functionality.
- **`src/distributions.rs`**: Target distributions (e.g., multivariate Gaussians) and proposal distributions.
- **`src/metropolis_hastings.rs`**: The Metropolis-Hastings algorithm implementation.
- **`src/gibbs.rs`**: The Gibbs sampling algorithm implementation.
- **`examples/demo.rs`**: Example usage demonstrating 2D Gaussian sampling and plotting.

## Usage (Local)

1. **Build** (Library + Demo):

   ```sh
   cargo build --release
   ```

2. **Run the Demo**:
   ```sh
   cargo run --release --bin gauss_mh
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
