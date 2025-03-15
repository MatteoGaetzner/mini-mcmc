/*!
# Metropolis–Hastings Sampler

This module implements a generic Metropolis–Hastings sampler that can work with any
target distribution `D` and proposal distribution `Q` that implement the corresponding
traits [`Target`] and [`Proposal`]. The sampler runs multiple independent Markov chains in parallel,
each initialized with the same starting state. A global seed is used to ensure reproducibility,
and each chain gets a unique seed by adding its index to the global seed.

## Overview

- **Target Distribution (`D`)**: Provides the (unnormalized) log-density for states via the [`Target`] trait.
- **Proposal Distribution (`Q`)**: Generates candidate states and computes the proposal density via the [`Proposal`] trait.
- **Parallel Chains**: The sampler maintains a vector of [`MHMarkovChain`] instances, each evolving independently.
- **Reproducibility**: The method `set_seed` assigns a unique seed to each chain based on a given global seed.

## Example Usage

```rust
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
use mini_mcmc::metropolis_hastings::MetropolisHastings;
use ndarray::{arr1, arr2};

// Define a 2D Gaussian target distribution with full covariance
let target = Gaussian2D {
    mean: arr1(&[0.0, 0.0]),
    cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
};

// Define an isotropic Gaussian proposal distribution (for any dimension)
let proposal = IsotropicGaussian::new(1.0);

// Starting state for all chains
let initial_state = [0.0, 0.0];

// Create a sampler with 1 chain
let mh = MetropolisHastings::new(target, proposal, &initial_state, 1);

// Check that one chain was created
assert_eq!(mh.chains.len(), 1);
```

See also the documentation for [`MHMarkovChain`] and the methods below.
*/

use num_traits::Float;
use rand::prelude::*;
use std::marker::{PhantomData, Send};

use crate::core::{HasChains, MarkovChain};
use crate::distributions::{Proposal, Target};

/**
The Metropolis–Hastings sampler generates samples from a target distribution by
using a proposal distribution to propose candidate moves and then accepting or rejecting
these moves using the Metropolis–Hastings acceptance criterion.

# Type Parameters
- `S`: The element type for the state (typically a floating-point type).
- `T`: The floating-point type (e.g. `f32` or `f64`).
- `D`: The target distribution type. Must implement [`Target`].
- `Q`: The proposal distribution type. Must implement [`Proposal`].

The sampler maintains multiple independent Markov chains (each represented by [`MHMarkovChain`])
that are run in parallel. A global random seed is provided, and each chain’s RNG is seeded by
adding the chain’s index to the global seed, ensuring reproducibility.

# Examples

```rust
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
use mini_mcmc::metropolis_hastings::MetropolisHastings;
use ndarray::{arr1, arr2};

let target = Gaussian2D {
    mean: arr1(&[0.0, 0.0]),
    cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
};
let proposal = IsotropicGaussian::new(1.0);
let initial_state = [0.0, 0.0];
let mh = MetropolisHastings::new(target, proposal, &initial_state, 1);
assert_eq!(mh.chains.len(), 1);
```
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetropolisHastings<S: Clone, T: Float, D: Clone, Q: Clone> {
    /// The target distribution we want to sample from.
    pub target: D,
    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,
    /// The vector of independent Markov chains.
    pub chains: Vec<MHMarkovChain<S, T, D, Q>>,
    /// The global random seed.
    pub seed: u64,
}

/// A single Markov chain for the Metropolis–Hastings algorithm.
///
/// Each chain stores its own copy of the target and proposal distributions,  
/// maintains its current state, and uses a chain-specific random number generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MHMarkovChain<S, T, D, Q> {
    /// The target distribution to sample from.
    pub target: D,
    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,
    /// The current state of the chain.
    pub current_state: Vec<S>,
    /// The chain-specific random seed.
    pub seed: u64,
    /// The random number generator for this chain.
    pub rng: SmallRng,
    phantom: PhantomData<T>,
}

impl<S, T, D, Q> MetropolisHastings<S, T, D, Q>
where
    D: Target<S, T> + std::clone::Clone + Send,
    Q: Proposal<S, T> + std::clone::Clone + Send,
    T: Float + Send,
    S: Clone + std::cmp::PartialEq + Send + num_traits::Zero + std::fmt::Debug + 'static,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /**
    Constructs a new Metropolis-Hastings sampler with a given target and proposal,
    initializing each chain at `initial_state` and creating `n_chains` parallel chains.

    # Arguments

    * `target` - The target distribution from which to sample.
    * `proposal` - The proposal distribution used to generate candidate states.
    * `initial_state` - The starting state for all chains.
    * `n_chains` - The number of parallel Markov chains to run.

    # Examples

    ```rust
    use mini_mcmc::metropolis_hastings::MetropolisHastings;
    use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use ndarray::{arr1, arr2};

    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let initial_state = [0.0, 0.0];
    let mh = MetropolisHastings::new(target, proposal, &initial_state, 1);
    assert_eq!(mh.chains.len(), 1);
    ```
    */
    pub fn new(target: D, proposal: Q, initial_state: &[S], n_chains: usize) -> Self {
        let chains = (0..n_chains)
            .map(|_| MHMarkovChain::new(target.clone(), proposal.clone(), initial_state))
            .collect();
        let seed = thread_rng().gen::<u64>();

        Self {
            target,
            proposal,
            chains,
            seed,
        }
    }

    /**
    Sets a new global seed and updates the seed for each chain accordingly.

    Each chain receives a unique seed calculated as `seed + i`, where `i` is the chain index.
    This method ensures reproducibility across runs and parallel chains.

    # Arguments

    * `seed` - The new global seed value.

    # Examples

    ```rust
    use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use mini_mcmc::metropolis_hastings::MetropolisHastings;
    use ndarray::{arr1, arr2};

    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let initial_state = [0.0, 0.0];
    let mh = MetropolisHastings::new(target, proposal, &initial_state, 2).set_seed(42);
    assert_eq!(mh.chains[0].seed, 42);
    assert_eq!(mh.chains[1].seed, 43);
    ```
    */
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        for (i, chain) in self.chains.iter_mut().enumerate() {
            let chain_seed = seed + i as u64;
            chain.seed = chain_seed;
            chain.rng = SmallRng::seed_from_u64(chain_seed)
        }
        self
    }
}

impl<S, T, D, Q> HasChains<S> for MetropolisHastings<S, T, D, Q>
where
    D: Target<S, T> + Clone + Send,
    Q: Proposal<S, T> + Clone + Send,
    T: Float + Send,
    S: Clone + PartialEq + Send + num_traits::Zero + std::fmt::Debug + 'static,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /// The concrete chain type used by the sampler.
    type Chain = MHMarkovChain<S, T, D, Q>;

    /// Returns a mutable reference to the internal vector of chains.
    ///
    /// This method allows external code to access and, if needed, modify the vector of  
    /// chains. For example, you may inspect or update individual chains using this reference.
    fn chains_mut(&mut self) -> &mut Vec<Self::Chain> {
        &mut self.chains
    }
}

impl<S, T, D, Q> MHMarkovChain<S, T, D, Q>
where
    D: Target<S, T> + Clone,
    Q: Proposal<S, T> + Clone,
    S: Clone + std::cmp::PartialEq + num_traits::Zero,
    T: Float,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /**
    Creates a new Metropolis–Hastings chain.

    # Arguments
    * `target` - The target distribution.
    * `proposal` - The proposal distribution.
    * `initial_state` - The starting state for the chain.

    # Examples

    ```rust
    use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use mini_mcmc::metropolis_hastings::MHMarkovChain;
    use ndarray::{arr1, arr2};

    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let chain = MHMarkovChain::new(target, proposal, &[0.0, 0.0]);
    assert_eq!(chain.current_state, vec![0.0, 0.0]);
    ```
    */
    pub fn new(target: D, proposal: Q, initial_state: &[S]) -> Self {
        let seed = thread_rng().gen::<u64>();
        Self {
            target,
            proposal,
            current_state: initial_state.to_vec(),
            seed,
            rng: SmallRng::seed_from_u64(seed),
            phantom: PhantomData,
        }
    }
}

impl<T, F, D, Q> MarkovChain<T> for MHMarkovChain<T, F, D, Q>
where
    D: Target<T, F> + Clone,
    Q: Proposal<T, F> + Clone,
    T: Clone + PartialEq + num_traits::Zero,
    F: Float,
    rand_distr::Standard: rand_distr::Distribution<F>,
{
    /**
    Performs one Metropolis–Hastings update step.

    A new candidate state is proposed using the proposal distribution.
    The unnormalized log-density of the current and proposed states is computed,
    along with the corresponding proposal densities. The acceptance ratio in log-space
    is calculated as:

    \[
    \log \alpha = \left[\log p(\text{proposed}) + \log q(\text{current} \mid \text{proposed})\right]
                  - \left[\log p(\text{current}) + \log q(\text{proposed} \mid \text{current})\right]
    \]

    A uniform random number is drawn, and if \(\log(\text{Uniform}(0,1))\) is less than
    \(\log \alpha\), the proposed state is accepted. Otherwise, the current state is retained.

    The method returns a reference to the updated state.

    # Examples

    ```rust
    use mini_mcmc::core::MarkovChain;
    use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use mini_mcmc::metropolis_hastings::MHMarkovChain;
    use ndarray::{arr1, arr2};

    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let mut chain = MHMarkovChain::new(target, proposal, &[0.0, 0.0]);
    let new_state = chain.step();
    assert_eq!(new_state.len(), 2);
    ```
    */
    fn step(&mut self) -> &Vec<T> {
        let proposed: Vec<T> = self.proposal.sample(&self.current_state);
        let current_lp = self.target.unnorm_log_prob(&self.current_state);
        let proposed_lp = self.target.unnorm_log_prob(&proposed);
        let log_q_forward = self.proposal.log_prob(&self.current_state, &proposed);
        let log_q_backward = self.proposal.log_prob(&proposed, &self.current_state);
        let log_accept_ratio = (proposed_lp + log_q_backward) - (current_lp + log_q_forward);
        let u: F = self.rng.gen();
        if log_accept_ratio > u.ln() {
            self.current_state = proposed;
        }
        &self.current_state
    }

    /// Returns a reference to the current state of the chain.
    fn current_state(&self) -> &Vec<T> {
        &self.current_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ChainRunner; // or run_progress, etc.
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, Axis};
    use ndarray_stats::CorrelationExt;

    use crate::distributions::{Gaussian2D, IsotropicGaussian};

    /// Common test harness for checking that samples from a 2D Gaussian match
    /// the true mean and covariance within floating-point tolerance.
    ///
    /// - `n_chains`: number of parallel chains
    /// - `use_progress`: whether to call `run_progress` instead of `run`
    fn run_gaussian_2d_test(sample_size: usize, n_chains: usize, use_progress: bool) {
        const BURNIN: usize = 2_000;
        const SEED: u64 = 42;

        // Target distribution
        let target = Gaussian2D {
            mean: arr1(&[0.0, 1.0]),
            cov: arr2(&[[4.0, 2.0], [2.0, 3.0]]),
        };

        // Build the sampler
        let initial_state = [0.0, 0.0];
        let proposal = IsotropicGaussian::new(1.0).set_seed(SEED);
        let mut mh = MetropolisHastings::new(target.clone(), proposal, &initial_state, n_chains)
            .set_seed(SEED);

        // Generate samples
        let samples = if use_progress {
            mh.run_progress(sample_size / n_chains, BURNIN).unwrap()
        } else {
            mh.run(sample_size / n_chains, BURNIN).unwrap()
        };

        // Reshape samples into a [sample_size, 2] array
        let stacked = samples
            .into_shape_with_order((sample_size, 2))
            .expect("Failed to reshape samples");

        // Check that mean and covariance match the target distribution
        let mean = stacked.mean_axis(Axis(0)).unwrap();
        let cov = stacked.t().cov(1.0).unwrap();
        assert_abs_diff_eq!(mean, target.mean, epsilon = 0.3); // or tighter threshold
        assert_abs_diff_eq!(cov, target.cov, epsilon = 0.5);
    }

    #[test]
    fn test_single_1_chain() {
        run_gaussian_2d_test(10_000, 1, false);
    }

    #[test]
    fn test_4_chains() {
        run_gaussian_2d_test(40_000, 4, false);
    }

    #[test]
    fn test_4_chains_long() {
        run_gaussian_2d_test(800_000, 4, false);
    }

    #[test]
    fn test_progress_1_chain() {
        run_gaussian_2d_test(10_000, 1, true);
    }

    #[test]
    fn test_progress_4_chains() {
        run_gaussian_2d_test(40_000, 4, true);
    }

    #[test]
    fn test_progress_4_chains_long() {
        run_gaussian_2d_test(800_000, 4, true);
    }

    #[test]
    #[ignore = "Slow test: run only when explicitly requested"]
    fn test_progress_16_chains_long() {
        run_gaussian_2d_test(8_000_000, 16, true);
    }

    /// This test remains separate because it's exercising the "example usage"
    /// scenario from the docs rather than checking numeric correctness.
    #[test]
    fn readme_test() {
        let target = Gaussian2D {
            mean: arr1(&[0.0, 0.0]),
            cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        };
        let proposal = IsotropicGaussian::new(1.0);
        let initial_state = [0.0, 0.0];

        // Create a MH sampler with 4 parallel chains
        let mut mh = MetropolisHastings::new(target, proposal, &initial_state, 4);

        // Run the sampler for 1100 steps, discarding the first 100 as burn-in
        let samples = mh.run(1000, 100).unwrap();

        // We should have 900 * 4 = 3600 samples
        assert_eq!(samples.shape()[0], 4);
        assert_eq!(samples.shape()[1], 1000);
    }
}
