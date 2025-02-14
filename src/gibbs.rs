use num_traits::Zero;
use rand::rngs::SmallRng;
use rand::{thread_rng, Rng, SeedableRng};

use nalgebra::Scalar;

use crate::core::{HasChains, MarkovChain};
use crate::distributions::Conditional;

pub struct GibbsMarkovChain<S, D>
where
    D: Conditional<S>,
{
    /// The distribution that provides conditional samples.
    pub target: D,

    /// Current state of the Markov chain.
    pub current_state: Vec<S>,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// RNG for this chain.
    pub rng: SmallRng,
}

impl<S, D> GibbsMarkovChain<S, D>
where
    D: Conditional<S> + Clone,
    S: Scalar + Zero,
{
    /// Creates a new chain with a given target distribution and initial state.
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
    /// Performs one full Gibbs sweep:
    ///   For each coordinate i in [0..dim), sample coordinate i
    ///   conditional on the others.
    fn step(&mut self) -> &std::vec::Vec<S> {
        (0..self.current_state.len())
            .for_each(|i| self.current_state[i] = self.target.sample(i, &self.current_state));
        &self.current_state
    }

    fn current_state(&self) -> &Vec<S> {
        &self.current_state
    }
}

pub struct GibbsSampler<S, D: Conditional<S>> {
    pub target: D,
    pub chains: Vec<GibbsMarkovChain<S, D>>,
    pub seed: u64,
}

impl<S, D> GibbsSampler<S, D>
where
    D: Conditional<S> + Clone + Send + Sync,
    S: Clone + Send + 'static + std::fmt::Debug + std::cmp::PartialEq + num_traits::Zero,
{
    /// Creates a new Gibbs sampler with `n_chains` parallel chains,
    /// all starting from `initial_state`.
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

    /// Sets a new seed, and updates the chains accordingly.
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
    // Define the concrete type of `Chain`:
    type Chain = GibbsMarkovChain<S, D>;

    fn chains_mut(&mut self) -> &mut Vec<Self::Chain> {
        &mut self.chains
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ChainRunner;
    use rand::RngCore;
    use rand_distr::Normal;
    use std::f64::consts::PI;

    /// A dummy conditional distribution that always returns the same constant value.
    #[derive(Clone)]
    struct ConstantConditional {
        c: f64,
    }

    impl Conditional<f64> for ConstantConditional {
        fn sample(&self, _i: usize, _given: &[f64]) -> f64 {
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
        fn sample(&self, i: usize, given: &[f64]) -> f64 {
            // Our state is [x, z], where z is 0.0 or 1.0.
            if i == 0 {
                // Sample x conditionally on z.
                let z = given[1];
                if z < 0.5 {
                    // Use mode 0: N(mu0, sigma0^2)
                    let normal = Normal::new(self.mu0, self.sigma0).unwrap();
                    thread_rng().sample(normal)
                } else {
                    let normal = Normal::new(self.mu1, self.sigma1).unwrap();
                    thread_rng().sample(normal)
                }
            } else if i == 1 {
                // Sample z conditionally on x.
                let x = given[0];
                let p0 = self.pi0 * MixtureConditional::normal_pdf(x, self.mu0, self.sigma0);
                let p1 =
                    (1.0 - self.pi0) * MixtureConditional::normal_pdf(x, self.mu1, self.sigma1);
                let total = p0 + p1;
                let prob_z1 = if total > 0.0 { p1 / total } else { 0.5 };
                if thread_rng().gen::<f64>() < prob_z1 {
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

    /// Test the run_with_progress method on the GibbsSampler.
    #[test]
    fn test_gibbs_sampler_run_with_progress() {
        let constant = 10.0;
        let conditional = ConstantConditional { c: constant };
        let initial_state = [0.0, 0.0];
        let mut sampler = GibbsSampler::new(conditional, &initial_state, 2);
        // Run with progress for 20 steps and discard the first 5.
        let samples = sampler.run_with_progress(20, 5);
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
            -42.0,                   // mu0
            69.0,                    // sigma0
            1.0,                     // mu1
            2.0,                     // sigma1
            0.123,                   // pi0
            3,                       // n_chains
            50000,                   // n_steps
            5000,                    // burn_in
            thread_rng().next_u64(), // seed
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
}
