/*!
# Gibbs Sampling

This module implements Gibbs sampling for MCMC. In this context, the target distribution
is specified via a trait [`Conditional`], which provides the full conditional distribution for
each coordinate of the state. The module defines:

- **`GibbsMarkovChain<S, D>`**: A single chain that performs a full Gibbs sweep (updating each coordinate in turn).
- **`GibbsSampler<S, D>`**: A sampler that maintains multiple parallel Gibbs chains, each initialized with
  the same starting state. A global seed is provided and each chain is assigned a unique seed.

The [`MarkovChain`] trait is implemented for `GibbsMarkovChain` so that generic chain-running
functions (e.g. via the [`ChainRunner`] extension) work with Gibbs chains.
*/

use num_traits::Zero;
use rand::rngs::SmallRng;
use rand::{thread_rng, Rng, SeedableRng};

use nalgebra::Scalar;

use crate::core::{HasChains, MarkovChain};
use crate::distributions::Conditional;

/// A single Gibbs sampling chain.
///
/// This chain updates its state by performing a full Gibbs sweep: for each coordinate `i` in  
/// the state vector, it samples a new value from the conditional distribution given the current  
/// values of all other coordinates. The conditional distribution is provided by the target  
/// via the [`Conditional`] trait.
///
/// # Type Parameters
/// - `S`: The type of each element in the state (typically a floating-point type).
/// - `D`: The type of the conditional distribution; must implement [`Conditional<S>`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GibbsMarkovChain<S, D>
where
    D: Conditional<S>,
{
    /// The conditional distribution that provides samples for each coordinate.
    pub target: D,
    /// The current state of the chain.
    pub current_state: Vec<S>,
    /// The random seed used for reproducibility.
    pub seed: u64,
    /// The chain-specific random number generator.
    pub rng: SmallRng,
}

impl<S, D> GibbsMarkovChain<S, D>
where
    D: Conditional<S> + Clone,
    S: Scalar + Zero,
{
    /// Creates a new Gibbs sampling chain.
    ///
    /// # Arguments
    ///
    /// * `target` - The conditional distribution that provides samples for each coordinate.
    /// * `initial_state` - The initial state vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mini_mcmc::gibbs::GibbsMarkovChain;
    /// use mini_mcmc::distributions::Conditional;
    /// use nalgebra::Scalar;
    ///
    /// // For example, a dummy conditional that always returns 1.0:
    /// #[derive(Clone)]
    /// struct OneConditional;
    /// impl Conditional<f64> for OneConditional {
    ///     fn sample(&mut self, _i: usize, _given: &[f64]) -> f64 { 1.0 }
    /// }
    ///
    /// let chain = GibbsMarkovChain::new(OneConditional, &[0.0, 0.0]);
    /// assert_eq!(chain.current_state, vec![0.0, 0.0]);
    /// ```
    pub fn new(target: D, initial_state: &[S]) -> Self {
        let seed = rand::thread_rng().gen::<u64>();
        Self {
            target,
            current_state: initial_state.to_vec(),
            seed,
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl<S, D: Conditional<S>> MarkovChain<S> for GibbsMarkovChain<S, D> {
    /// Performs one full Gibbs sweep.
    ///
    /// For each coordinate `i` in `0..current_state.len()`, a new value is sampled from  
    /// the conditional distribution (provided by `target.sample(i, &current_state)`) and  
    /// the state is updated. Returns a reference to the updated state.
    fn step(&mut self) -> &Vec<S> {
        (0..self.current_state.len())
            .for_each(|i| self.current_state[i] = self.target.sample(i, &self.current_state));
        &self.current_state
    }

    /// Returns a reference to the current state of the chain.
    fn current_state(&self) -> &Vec<S> {
        &self.current_state
    }
}

/// A Gibbs sampler that runs multiple parallel chains.
///
/// The sampler creates several independent Gibbs chains (of type [`GibbsMarkovChain`]),  
/// all initialized with the same starting state. A global seed is used for reproducibility,  
/// and each chain is assigned a unique seed by adding its index to the global seed.
///
/// # Type Parameters
/// - `S`: The type of each element in the state (typically a floating-point type).
/// - `D`: The type of the conditional distribution; must implement [`Conditional<S>`].
pub struct GibbsSampler<S, D: Conditional<S>> {
    /// The conditional distribution used by all chains.
    pub target: D,
    /// The vector of Gibbs chains.
    pub chains: Vec<GibbsMarkovChain<S, D>>,
    /// The global seed for the sampler.
    pub seed: u64,
}

impl<S, D> GibbsSampler<S, D>
where
    D: Conditional<S> + Clone + Send + Sync,
    S: Clone + Send + 'static + std::fmt::Debug + std::cmp::PartialEq + num_traits::Zero,
{
    /// Creates a new Gibbs sampler with a specified number of parallel chains.
    ///
    /// All chains are initialized with the same `initial_state` and the provided conditional  
    /// distribution.
    ///
    /// # Arguments
    ///
    /// * `target` - The conditional distribution that specifies the full conditionals for each coordinate.
    /// * `initial_state` - The starting state vector for every chain.
    /// * `n_chains` - The number of chains to run in parallel.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mini_mcmc::gibbs::GibbsSampler;
    /// use mini_mcmc::distributions::Conditional;
    ///
    /// #[derive(Clone)]
    /// struct DummyConditional;
    /// impl Conditional<f64> for DummyConditional {
    ///     fn sample(&mut self, _i: usize, _given: &[f64]) -> f64 { 0.0 }
    /// }
    ///
    /// let sampler = GibbsSampler::new(DummyConditional, &[0.0, 0.0], 4);
    /// assert_eq!(sampler.chains.len(), 4);
    /// ```
    pub fn new(target: D, initial_state: &[S], n_chains: usize) -> Self {
        let seed = thread_rng().gen::<u64>();
        let chains = (0..n_chains)
            .map(|_| GibbsMarkovChain::new(target.clone(), initial_state))
            .collect();

        Self {
            target,
            chains,
            seed,
        }
    }

    /// Sets a new seed for the sampler and updates the seed for each chain.
    ///
    /// Each chain's seed is updated to `seed + i`, where `i` is the chain index.
    ///
    /// # Arguments
    ///
    /// * `seed` - The new global seed.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        for (i, chain) in self.chains.iter_mut().enumerate() {
            let chain_seed = seed + i as u64;
            chain.seed = chain_seed;
            chain.rng = SmallRng::seed_from_u64(chain_seed);
        }
        self
    }
}

impl<S, D> HasChains<S> for GibbsSampler<S, D>
where
    D: Conditional<S> + Clone + Send + Sync,
    S: std::marker::Send,
{
    /// The concrete chain type used by this sampler.
    type Chain = GibbsMarkovChain<S, D>;

    /// Returns a mutable reference to the internal vector of Gibbs chains.
    ///
    /// This method is used by generic chain-running utilities (such as those in [`ChainRunner`])
    /// to access and manage the sampler's chains.
    fn chains_mut(&mut self) -> &mut Vec<Self::Chain> {
        &mut self.chains
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ChainRunner;
    use rand_distr::Normal;
    use std::f64::consts::PI;

    /// A dummy conditional distribution that always returns the same constant value.
    #[derive(Clone)]
    struct ConstantConditional {
        c: f64,
    }

    impl Conditional<f64> for ConstantConditional {
        fn sample(&mut self, _i: usize, _given: &[f64]) -> f64 {
            self.c
        }
    }

    /// A conditional distribution for a 2D state [x, z] that targets a two-component Gaussian mixture.
    /// The parameters:
    ///   - If z == 0: x ~ N(mu0, sigma0^2)
    ///   - If z == 1: x ~ N(mu1, sigma1^2)
    ///   - The latent z is updated by computing the conditional probabilities:
    ///       p(z=0|x) ∝ π0 * N(x; mu0, sigma0^2)
    ///       p(z=1|x) ∝ (1-π0) * N(x; mu1, sigma1^2)
    #[derive(Clone)]
    struct MixtureConditional {
        mu0: f64,
        sigma0: f64,
        mu1: f64,
        sigma1: f64,
        pi0: f64, // mixing proportion for mode 0 (assume 0 < pi0 < 1, and mode 1 has weight 1 - pi0)
        rng: SmallRng,
    }

    impl MixtureConditional {
        /// A simple implementation of the normal density function.
        fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
            let var = sigma * sigma;
            let coeff = 1.0 / ((2.0 * PI * var).sqrt());
            let exp_val = (-((x - mu).powi(2)) / (2.0 * var)).exp();
            coeff * exp_val
        }
    }

    impl Conditional<f64> for MixtureConditional {
        fn sample(&mut self, i: usize, given: &[f64]) -> f64 {
            // Our state is [x, z], where z is 0.0 or 1.0.
            if i == 0 {
                // Sample x conditionally on z.
                let z = given[1];
                if z < 0.5 {
                    // Use mode 0: N(mu0, sigma0^2)
                    let normal = Normal::new(self.mu0, self.sigma0).unwrap();
                    self.rng.sample(normal)
                } else {
                    let normal = Normal::new(self.mu1, self.sigma1).unwrap();
                    self.rng.sample(normal)
                }
            } else if i == 1 {
                // Sample z conditionally on x.
                let x = given[0];
                let p0 = self.pi0 * MixtureConditional::normal_pdf(x, self.mu0, self.sigma0);
                let p1 =
                    (1.0 - self.pi0) * MixtureConditional::normal_pdf(x, self.mu1, self.sigma1);
                let total = p0 + p1;
                let prob_z1 = if total > 0.0 { p1 / total } else { 0.5 };
                if self.rng.gen::<f64>() < prob_z1 {
                    1.0
                } else {
                    0.0
                }
            } else {
                panic!("Invalid coordinate index in MixtureConditional");
            }
        }
    }

    /// Test that a single GibbsMarkovChain updates its state correctly.
    #[test]
    fn test_gibbs_chain_step() {
        // Create a conditional that always returns 7.0.
        let conditional = ConstantConditional { c: 7.0 };
        // Start with a 3-dimensional state.
        let initial_state = [0.0, 0.0, 0.0];
        let mut chain = GibbsMarkovChain::new(conditional, &initial_state);

        // After one Gibbs sweep, every coordinate should be updated to 7.0.
        chain.step();
        for &x in chain.current_state().iter() {
            assert!((x - 7.0).abs() < f64::EPSILON, "Expected 7.0, got {}", x);
        }
    }

    /// Test that the GibbsSampler (which runs multiple chains) converges to the constant value.
    #[test]
    fn test_gibbs_sampler_run() {
        let constant = 42.0;
        let conditional = ConstantConditional { c: constant };
        // 2-dimensional state.
        let initial_state = [0.0, 0.0];
        // Create a sampler with 4 chains.
        let mut sampler = GibbsSampler::new(conditional, &initial_state, 4);
        // Run each chain for 10 steps and discard the first 5 as burn-in.
        let samples = sampler.run(10, 5);
        // Each chain should have 10 - 5 = 5 rows and 2 columns.
        for (chain_idx, chain_samples) in samples.iter().enumerate() {
            assert_eq!(
                chain_samples.nrows(),
                5,
                "Chain {} row count mismatch",
                chain_idx
            );
            assert_eq!(
                chain_samples.ncols(),
                2,
                "Chain {} column count mismatch",
                chain_idx
            );
            // Check that the final (last) row is (approximately) equal to [42.0, 42.0].
            let last_row = chain_samples.row(chain_samples.nrows() - 1);
            for (j, &value) in last_row.iter().enumerate() {
                assert!(
                    (value - constant).abs() < f64::EPSILON,
                    "Chain {}, coordinate {} expected {}, got {}",
                    chain_idx,
                    j,
                    constant,
                    value
                );
            }
        }
    }

    /// Test the run_progress method on the GibbsSampler.
    #[test]
    fn test_gibbs_sampler_run_progress() {
        let constant = 10.0;
        let conditional = ConstantConditional { c: constant };
        let initial_state = [0.0, 0.0];
        let mut sampler = GibbsSampler::new(conditional, &initial_state, 2);
        // Run with progress for 20 steps and discard the first 5.
        let samples = sampler.run_progress(20, 5);
        for (chain_idx, chain_samples) in samples.iter().enumerate() {
            assert_eq!(
                chain_samples.nrows(),
                15,
                "Chain {} row count mismatch",
                chain_idx
            );
            let last_row = chain_samples.row(chain_samples.nrows() - 1);
            for (j, &value) in last_row.iter().enumerate() {
                assert!(
                    (value - constant).abs() < f64::EPSILON,
                    "Chain {}, coordinate {} expected {}, got {}",
                    chain_idx,
                    j,
                    constant,
                    value
                );
            }
        }
    }

    /// Helper function that runs a GibbsSampler for a mixture distribution
    /// and returns (theoretical_mean, theoretical_variance, sample_mean, sample_variance)
    #[allow(clippy::too_many_arguments)]
    fn run_mixture_simulation(
        mu0: f64,
        sigma0: f64,
        mu1: f64,
        sigma1: f64,
        pi0: f64,
        n_chains: usize,
        n_steps: usize,
        burn_in: usize,
        seed: u64,
    ) -> (f64, f64, f64, f64) {
        // Compute theoretical marginal mean and variance for x.
        let theoretical_mean = pi0 * mu0 + (1.0 - pi0) * mu1;
        let theoretical_variance = pi0 * (sigma0.powi(2) + (mu0 - theoretical_mean).powi(2))
            + (1.0 - pi0) * (sigma1.powi(2) + (mu1 - theoretical_mean).powi(2));

        let conditional = MixtureConditional {
            mu0,
            sigma0,
            mu1,
            sigma1,
            pi0,
            rng: SmallRng::seed_from_u64(seed),
        };
        let initial_state = [0.0, 0.0];
        let mut sampler = GibbsSampler::new(conditional, &initial_state, n_chains).set_seed(seed);
        let samples = sampler.run(n_steps, burn_in);

        // Collect all x-values from all chains.
        let mut all_x = Vec::new();
        for mat in samples {
            for i in 0..mat.nrows() {
                let row = mat.row(i);
                all_x.push(row[0]);
            }
        }
        let n = all_x.len() as f64;
        let sample_mean: f64 = all_x.iter().sum::<f64>() / n;
        let sample_variance: f64 = all_x
            .iter()
            .map(|&x| (x - sample_mean).powi(2))
            .sum::<f64>()
            / n;
        (
            theoretical_mean,
            theoretical_variance,
            sample_mean,
            sample_variance,
        )
    }

    /// Test the GibbsSampler on a two-component Gaussian mixture (set 1).
    #[test]
    fn test_gibbs_sampler_mixture_1() {
        let (theo_mean, theo_var, sample_mean, sample_var) = run_mixture_simulation(
            -2.0, // mu0
            1.0,  // sigma0
            3.0,  // mu1
            1.5,  // sigma1
            0.5,  // pi0
            3,    // n_chains
            5000, // n_steps
            1000, // burn_in
            42,   // seed
        );
        println!("Mixture 1:");
        println!("Theoretical mean: {}", theo_mean);
        println!("Empirical mean: {}", sample_mean);
        println!("Theoretical variance: {}", theo_var);
        println!("Empirical variance: {}", sample_var);

        assert!(
            (sample_mean - theo_mean).abs() < 0.5,
            "Empirical mean {} deviates too much from theoretical {}",
            sample_mean,
            theo_mean
        );
        assert!(
            (sample_var - theo_var).abs() < 1.5,
            "Empirical variance {} deviates too much from theoretical {}",
            sample_var,
            theo_var
        );
    }

    /// Test the GibbsSampler on a two-component Gaussian mixture (set 2).
    #[test]
    fn test_gibbs_sampler_mixture_2() {
        let (theo_mean, theo_var, sample_mean, sample_var) = run_mixture_simulation(
            -42.0,  // mu0
            69.0,   // sigma0
            1.0,    // mu1
            2.0,    // sigma1
            0.123,  // pi0
            3,      // n_chains
            100000, // n_steps
            10000,  // burn_in
            42,     // seed
        );
        println!("Mixture 2:");
        println!("Theoretical mean: {}", theo_mean);
        println!("Empirical mean: {}", sample_mean);
        println!("Theoretical variance: {}", theo_var);
        println!("Empirical variance: {}", sample_var);

        assert!(
            (sample_mean - theo_mean).abs() < theo_mean.abs() / 10.0,
            "Empirical mean {} deviates too much from theoretical {}",
            sample_mean,
            theo_mean
        );
        assert!(
            (sample_var - theo_var).abs() < theo_var.abs() / 10.0,
            "Empirical variance {} deviates too much from theoretical {}",
            sample_var,
            theo_var
        );
    }

    #[test]
    fn test_chain_step_return_value() {
        // Use the constant conditional so we know exactly what each coordinate updates to.
        let conditional = ConstantConditional { c: 42.0 };
        // 3D initial state
        let initial_state = [0.0, 0.0, 0.0];
        let mut chain = GibbsMarkovChain::new(conditional, &initial_state);

        // Call step() and capture the return value
        let returned_ref = chain.step();

        // 1) Check that all coordinates have been updated to 42.0.
        for &val in returned_ref.iter() {
            assert!(
                (val - 42.0).abs() < f64::EPSILON,
                "Expected 42.0 after step, got {}",
                val
            );
        }

        // 2) Check that step() returns the same reference as chain.current_state().
        assert!(
            std::ptr::eq(returned_ref, chain.current_state()),
            "step() should return a reference to the chain's internal state"
        );
    }

    #[test]
    fn test_chain_current_state_return_value() {
        // Use a different constant for clarity
        let conditional = ConstantConditional { c: 13.0 };
        let initial_state = [1.0, 2.0, 3.0];
        let chain = GibbsMarkovChain::new(conditional, &initial_state);

        // Call current_state() and ensure it matches the chain's internal state.
        let state_ref = chain.current_state();

        // 1) Check length
        assert_eq!(
            state_ref.len(),
            initial_state.len(),
            "Expected the current_state() to have length {}",
            initial_state.len()
        );

        // 2) Check each coordinate
        for (i, &val) in state_ref.iter().enumerate() {
            assert!(
                (val - initial_state[i]).abs() < f64::EPSILON,
                "Expected coordinate {} to be {}, got {}",
                i,
                initial_state[i],
                val
            );
        }
    }

    #[test]
    fn test_has_chains_for_gibbs_sampler() {
        // Create a GibbsSampler with a constant conditional.
        let conditional = ConstantConditional { c: 42.0 };
        let initial_state = [0.0, 0.0];
        let mut sampler = GibbsSampler::new(conditional.clone(), &initial_state, 3);
        let original_len = sampler.chains.len();

        // Use chains_mut() to get a mutable reference to the internal chains.
        {
            let chains_mut = sampler.chains_mut();
            // Modify the first chain's first coordinate.
            if let Some(first_chain) = chains_mut.first_mut() {
                first_chain.current_state[0] = 100.0;
            }
        }
        // Now, check that the modification is visible via sampler.chains.
        assert_eq!(
            sampler.chains[0].current_state[0], 100.0,
            "Expected the first coordinate of the first chain to be updated to 100.0"
        );

        // Use chains_mut() again to push a new chain.
        {
            let chains_mut = sampler.chains_mut();
            chains_mut.push(GibbsMarkovChain::new(conditional, &initial_state));
        }
        // The new length should be original_len + 1.
        assert_eq!(
            sampler.chains.len(),
            original_len + 1,
            "Expected chains length to increase by 1"
        );
    }
}
